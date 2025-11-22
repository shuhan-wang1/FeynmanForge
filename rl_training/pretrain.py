"""
pretrain.py
Supervised pre-training for Feynman-GCPN

Teaches the model to build e+e- -> mu+mu- topology
using expert demonstrations.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from tqdm import tqdm
import numpy as np
import os

from models import FeynmanGCPN
from feynman_env import FeynmanDiagramEnv
from physics_engine import PhysicsConstants
from generate_expert import generate_expert_dataset


class ExpertDataset(Dataset):
    """Dataset wrapper for expert (state, action) pairs"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action = self.data[idx]
        return state, action


def collate_fn(batch):
    """
    Collate function for PyG Data objects

    Returns:
        states: Batched PyG Data
        actions: List of action dicts
    """
    states = [item[0] for item in batch]
    actions = [item[1] for item in batch]

    # Batch PyG Data objects
    batched_states = Batch.from_data_list(states)

    return batched_states, actions


def pretrain_model(
    num_episodes=500,
    batch_size=32,
    num_epochs=20,
    lr=1e-3,
    device='auto'
):
    """
    Pre-train the model using supervised learning on expert data

    Args:
        num_episodes: Number of expert trajectories to generate
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to use ('auto', 'cuda', or 'cpu')
    """
    # 1. Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print(f"Using device: {device}")

    # 2. Generate expert data
    print("\n" + "=" * 80)
    print("STEP 1: Generating Expert Data")
    print("=" * 80)

    expert_data = generate_expert_dataset(num_episodes=num_episodes)

    print(f"\nTotal training samples: {len(expert_data)}")

    # 3. Create dataset and dataloader
    dataset = ExpertDataset(expert_data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues with PyG
    )

    # 4. Initialize model
    print("\n" + "=" * 80)
    print("STEP 2: Initializing Model")
    print("=" * 80)

    num_particle_types = len(PhysicsConstants.get_all_particles()) + len(PhysicsConstants.BOSONS)

    model = FeynmanGCPN(
        node_input_dim=9,
        edge_input_dim=21,
        hidden_dim=128,
        num_mp_layers=3,
        num_action_types=5,
        num_particle_types=num_particle_types,
        max_vertices=10
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 6. Training loop
    print("\n" + "=" * 80)
    print("STEP 3: Training")
    print("=" * 80)

    model.train()
    best_acc = 0.0

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_actions = 0
        correct_vertices = 0
        correct_particles = 0
        total_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for states, actions in pbar:
            states = states.to(device)

            # Forward pass
            # Extract vertex states for each graph in batch
            batch_size_actual = len(actions)
            vertex_states_batch = [None] * batch_size_actual  # Placeholder

            outputs = model(states, vertex_states_batch, return_value=False)

            # Prepare targets
            target_types = torch.tensor(
                [a['action_type'] for a in actions],
                dtype=torch.long,
                device=device
            )
            target_vertices = torch.tensor(
                [a.get('vertex_idx', 0) for a in actions],
                dtype=torch.long,
                device=device
            )
            target_particles = torch.tensor(
                [a.get('particle_type', 0) for a in actions],
                dtype=torch.long,
                device=device
            )

            # Compute losses
            loss_type = criterion(outputs['action_type_logits'], target_types)
            loss_vertex = criterion(outputs['vertex_logits'], target_vertices)
            loss_particle = criterion(outputs['particle_logits'], target_particles)

            # Total loss (weighted sum)
            loss = loss_type + 0.5 * loss_vertex + 0.5 * loss_particle

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Statistics
            total_loss += loss.item()

            preds_type = outputs['action_type_logits'].argmax(dim=1)
            preds_vertex = outputs['vertex_logits'].argmax(dim=1)
            preds_particle = outputs['particle_logits'].argmax(dim=1)

            correct_actions += (preds_type == target_types).sum().item()
            correct_vertices += (preds_vertex == target_vertices).sum().item()
            correct_particles += (preds_particle == target_particles).sum().item()
            total_samples += batch_size_actual

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc_type': f'{correct_actions/total_samples:.2%}'
            })

        # Epoch summary
        avg_loss = total_loss / len(dataloader)
        acc_type = correct_actions / total_samples
        acc_vertex = correct_vertices / total_samples
        acc_particle = correct_particles / total_samples

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Action Type Acc: {acc_type:.2%}")
        print(f"  Vertex Acc: {acc_vertex:.2%}")
        print(f"  Particle Acc: {acc_particle:.2%}")

        # Save best model
        if acc_type > best_acc:
            best_acc = acc_type
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': acc_type,
            }, 'checkpoints/pretrained_model.pt')
            print(f"  ✅ New best model saved! (acc: {best_acc:.2%})")

    # 7. Final save
    print("\n" + "=" * 80)
    print("STEP 4: Saving Final Model")
    print("=" * 80)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': avg_loss,
        'final_accuracy': acc_type,
    }, 'checkpoints/pretrained_final.pt')

    print(f"✅ Pre-training complete!")
    print(f"   Best accuracy: {best_acc:.2%}")
    print(f"   Models saved:")
    print(f"     - checkpoints/pretrained_model.pt (best)")
    print(f"     - checkpoints/pretrained_final.pt (final)")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Pre-train Feynman-GCPN')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of expert trajectories')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cuda/cpu)')

    args = parser.parse_args()

    pretrain_model(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device
    )

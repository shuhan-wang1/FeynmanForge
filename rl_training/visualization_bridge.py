"""
Visualization Bridge: Exports diagrams to JavaScript-compatible format
Enables real-time monitoring using the existing Feynman Forge frontend
"""

import json
import os
from typing import List, Dict, Optional
from datetime import datetime
import torch

from feynman_env import FeynmanDiagramEnv
from models import FeynmanGCPN
from physics_engine import PhysicsConstants


class DiagramExporter:
    """
    Exports Feynman diagrams to JSON format compatible with feynman-logic.js
    """
    
    @staticmethod
    def env_to_shapes(env: FeynmanDiagramEnv) -> List[Dict]:
        """
        Convert environment state to shapes array for canvas-manager.js
        
        Returns:
            List of shape dictionaries matching the format:
            {
                "id": int,
                "type": "fermion" | "photon" | "boson_w" | "gluon" | "higgs",
                "p1": {"x": float, "y": float},
                "p2": {"x": float, "y": float},
                "props": {
                    "particleId": str,
                    "isAnti": bool,
                    "color": str | null,
                    "category": "fermion" | "boson",
                    "group": str
                }
            }
        """
        return env.get_diagram_json()
    
    @staticmethod
    def save_diagram(
        shapes: List[Dict],
        filepath: str = 'diagrams/current_best.json',
        metadata: Optional[Dict] = None
    ):
        """
        Save diagram to JSON file with optional metadata
        
        Args:
            shapes: List of shape dictionaries
            filepath: Path to save JSON file
            metadata: Optional metadata (episode, reward, etc.)
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'shapes': shapes
        }
        
        if metadata:
            output['metadata'] = metadata
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def export_training_sequence(
        diagrams: List[List[Dict]],
        rewards: List[float],
        filepath: str = 'diagrams/training_sequence.json'
    ):
        """
        Export a sequence of diagrams from training (for animation)
        
        Args:
            diagrams: List of shape arrays (one per episode)
            rewards: Corresponding rewards
            filepath: Output file path
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        sequence = []
        for i, (diagram, reward) in enumerate(zip(diagrams, rewards)):
            sequence.append({
                'episode': i,
                'reward': reward,
                'shapes': diagram,
                'timestamp': datetime.now().isoformat()
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sequence, f, indent=2)


class LiveMonitor:
    """
    Real-time monitoring callback for training
    Exports diagrams periodically for visualization
    """
    
    def __init__(
        self,
        export_interval: int = 10,
        export_dir: str = 'diagrams',
        max_history: int = 100
    ):
        """
        Args:
            export_interval: Export diagram every N episodes
            export_dir: Directory to save diagrams
            max_history: Maximum number of historical diagrams to keep
        """
        self.export_interval = export_interval
        self.export_dir = export_dir
        self.max_history = max_history
        
        self.episode_count = 0
        self.diagram_history = []
        self.reward_history = []
        
        os.makedirs(export_dir, exist_ok=True)
    
    def on_episode_end(
        self,
        env: FeynmanDiagramEnv,
        episode_reward: float,
        episode_length: int
    ):
        """
        Callback after each episode
        
        Args:
            env: The environment (with current diagram)
            episode_reward: Total reward for this episode
            episode_length: Number of steps
        """
        self.episode_count += 1
        
        # Export current diagram
        if self.episode_count % self.export_interval == 0:
            shapes = DiagramExporter.env_to_shapes(env)
            
            metadata = {
                'episode': self.episode_count,
                'reward': episode_reward,
                'length': episode_length,
                'initial_state': env.initial_particles,
                'final_state': env.final_particles
            }
            
            # Save current diagram
            DiagramExporter.save_diagram(
                shapes,
                filepath=os.path.join(self.export_dir, 'current_diagram.json'),
                metadata=metadata
            )
            
            # Add to history
            self.diagram_history.append(shapes)
            self.reward_history.append(episode_reward)
            
            # Trim history
            if len(self.diagram_history) > self.max_history:
                self.diagram_history = self.diagram_history[-self.max_history:]
                self.reward_history = self.reward_history[-self.max_history:]
            
            # Export sequence
            DiagramExporter.export_training_sequence(
                self.diagram_history,
                self.reward_history,
                filepath=os.path.join(self.export_dir, 'training_sequence.json')
            )
    
    def on_best_diagram(
        self,
        env: FeynmanDiagramEnv,
        reward: float
    ):
        """
        Callback when a new best diagram is found
        
        Args:
            env: Environment with the best diagram
            reward: Best reward achieved
        """
        shapes = DiagramExporter.env_to_shapes(env)
        
        metadata = {
            'episode': self.episode_count,
            'reward': reward,
            'initial_state': env.initial_particles,
            'final_state': env.final_particles,
            'is_best': True
        }
        
        DiagramExporter.save_diagram(
            shapes,
            filepath=os.path.join(self.export_dir, 'best_diagram.json'),
            metadata=metadata
        )
        
        print(f"  🏆 New best diagram! Reward: {reward:.2f}")


class DiagramEvaluator:
    """
    Evaluate trained model and export diagrams
    """
    
    def __init__(
        self,
        model: FeynmanGCPN,
        env: FeynmanDiagramEnv,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.env = env
        self.device = device
    
    def generate_diagram(
        self,
        max_steps: int = 50,
        deterministic: bool = True
    ) -> tuple[List[Dict], float]:
        """
        Generate a diagram using the trained model
        
        Args:
            max_steps: Maximum number of steps
            deterministic: Whether to use deterministic policy
            
        Returns:
            shapes: List of shape dictionaries
            total_reward: Total reward achieved
        """
        state, info = self.env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            state_device = state.to(self.device)
            
            with torch.no_grad():
                action = self.model.get_action(state_device, deterministic=deterministic)
            
            next_state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
            
            state = next_state
        
        shapes = DiagramExporter.env_to_shapes(self.env)
        
        return shapes, total_reward
    
    def evaluate_multiple(
        self,
        num_episodes: int = 10,
        output_dir: str = 'evaluation'
    ) -> Dict:
        """
        Generate multiple diagrams and save them
        
        Args:
            num_episodes: Number of diagrams to generate
            output_dir: Directory to save results
            
        Returns:
            Statistics dictionary
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_diagrams = []
        all_rewards = []
        
        for i in range(num_episodes):
            shapes, reward = self.generate_diagram(deterministic=True)
            all_diagrams.append(shapes)
            all_rewards.append(reward)
            
            # Save individual diagram
            DiagramExporter.save_diagram(
                shapes,
                filepath=os.path.join(output_dir, f'diagram_{i:03d}.json'),
                metadata={'episode': i, 'reward': reward}
            )
        
        # Save best
        best_idx = np.argmax(all_rewards)
        DiagramExporter.save_diagram(
            all_diagrams[best_idx],
            filepath=os.path.join(output_dir, 'best_diagram.json'),
            metadata={'episode': best_idx, 'reward': all_rewards[best_idx], 'is_best': True}
        )
        
        # Statistics
        stats = {
            'num_episodes': num_episodes,
            'mean_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'max_reward': float(np.max(all_rewards)),
            'min_reward': float(np.min(all_rewards)),
            'best_episode': int(best_idx)
        }
        
        # Save stats
        with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"Episodes: {num_episodes}")
        print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Best Reward: {stats['max_reward']:.2f}")
        print(f"Worst Reward: {stats['min_reward']:.2f}")
        print("=" * 60)
        
        return stats


def create_reaction_config(
    initial_particles: List[str],
    final_particles: List[str],
    output_file: str = 'diagrams/reaction_config.json'
):
    """
    Create a configuration file for the visualization frontend
    
    Args:
        initial_particles: List of initial particle IDs
        final_particles: List of final particle IDs
        output_file: Path to save config
    """
    config = {
        'reaction': {
            'initial': initial_particles,
            'final': final_particles,
            'latex': f"${' + '.join(initial_particles)} \\to {' + '.join(final_particles)}$"
        },
        'visualization': {
            'canvas_width': 800,
            'canvas_height': 600,
            'initial_x': 80,
            'final_x': 720,
            'auto_refresh': True,
            'refresh_interval': 2000  # ms
        }
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created reaction config: {output_file}")


if __name__ == '__main__':
    # Example: Export a dummy diagram
    from feynman_env import FeynmanDiagramEnv
    
    env = FeynmanDiagramEnv(
        initial_state=['e', 'e'],
        final_state=['mu', 'mu']
    )
    
    env.reset()
    
    # Perform a few actions
    env.step({'action_type': 1, 'vertex_idx': 0, 'particle_type': 3, 'target_vertex': 1})
    
    # Export
    shapes = DiagramExporter.env_to_shapes(env)
    DiagramExporter.save_diagram(shapes, 'diagrams/test_diagram.json')
    
    print("Test diagram exported to diagrams/test_diagram.json")
    print(json.dumps(shapes, indent=2))

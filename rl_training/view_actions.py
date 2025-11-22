#!/usr/bin/env python3
"""
Simple script to view action history from saved diagram JSON
Shows step-by-step what the model did to build the diagram
"""

import json
import sys
from pathlib import Path

def view_actions(json_path='diagrams/current_best.json'):
    """Load and display action history from diagram JSON"""

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {json_path}")
        print("Make sure training has run and saved diagrams!")
        return

    # Extract data
    metadata = data.get('metadata', {})
    actions = data.get('actions', [])
    shapes = data.get('shapes', [])

    print("=" * 80)
    print("🎯 FEYNMAN DIAGRAM CONSTRUCTION HISTORY")
    print("=" * 80)
    print(f"Reaction: {' + '.join(metadata.get('initial_state', []))} → {' + '.join(metadata.get('final_state', []))}")
    print(f"Episode: {metadata.get('episode', 0)}")
    print(f"Total Steps: {metadata.get('num_steps', 0)}")
    print(f"Final Shapes: {len(shapes)}")
    print("=" * 80)

    if not actions:
        print("\n⚠️  No actions recorded! Make sure you're running the latest code.")
        print("   Pull the latest commit: git pull origin claude/optimize-training-performance-...")
        return

    print(f"\n📋 ACTION TIMELINE ({len(actions)} steps):\n")

    for action in actions:
        step = action['step']
        action_type = action['action_type']
        success = action['success']
        reward = action['reward']

        # Status icon
        status = "✅" if success else "❌"

        # Reward color
        if reward > 1:
            reward_str = f"🟢 +{reward:.2f}"
        elif reward > 0:
            reward_str = f"🟡 +{reward:.2f}"
        elif reward > -1:
            reward_str = f"🟠 {reward:.2f}"
        else:
            reward_str = f"🔴 {reward:.2f}"

        print(f"Step {step}: {status} {action_type:<12}", end="")

        # Action details
        if action_type in ['CONNECT', 'MERGE']:
            v1 = action.get('vertex_idx', '?')
            v2 = action.get('target_vertex', '?')
            print(f" vertex {v1} → {v2}", end="")
        elif action_type == 'BRANCH':
            v = action.get('vertex_idx', '?')
            p = action.get('particle_type', '?')
            print(f" from vertex {v} (particle #{p})", end="")
        elif action_type == 'SET_TYPE':
            v = action.get('vertex_idx', '?')
            p = action.get('particle_type', '?')
            print(f" vertex {v} to particle #{p}", end="")

        # Graph growth
        vertices_before = action['num_vertices_before']
        vertices_after = action['num_vertices_after']
        if vertices_after > vertices_before:
            print(f" | Vertices: {vertices_before} → {vertices_after}", end="")

        print(f" | Reward: {reward_str}")

    print("\n" + "=" * 80)

    # Summary
    successful = sum(1 for a in actions if a['success'])
    failed = sum(1 for a in actions if not a['success'])
    total_reward = sum(a['reward'] for a in actions)

    print(f"📊 SUMMARY:")
    print(f"   Successful actions: {successful}/{len(actions)} ({successful/len(actions)*100:.1f}%)")
    print(f"   Failed actions: {failed}/{len(actions)} ({failed/len(actions)*100:.1f}%)")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Final vertices: {actions[-1]['num_vertices_after'] if actions else 0}")
    print("=" * 80)

if __name__ == '__main__':
    json_path = sys.argv[1] if len(sys.argv) > 1 else 'diagrams/current_best.json'
    view_actions(json_path)

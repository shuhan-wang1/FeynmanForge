"""
Quick Start Script for Feynman-GCPN V8
Runs a short training session to verify the setup
"""

import subprocess
import sys

def main():
    print("=" * 80)
    print("🚀 Feynman-GCPN V8 Quick Start")
    print("=" * 80)
    print()
    print("This will run a short training session (10k steps) to verify your setup.")
    print()

    # Run quick experiment
    cmd = [
        sys.executable,
        "run_experiment.py",
        "--quick",
        "--output", "quick_test_results"
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=sys.path[0])

    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("✓ Quick start completed successfully!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Check results in: quick_test_results/")
        print("2. Run full training: python run_experiment.py")
        print("3. Custom reaction: python run_experiment.py --reaction 'e+e_bar->mu+mu_bar'")
    else:
        print("\n" + "=" * 80)
        print("✗ Quick start failed. Please check the error messages above.")
        print("=" * 80)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())

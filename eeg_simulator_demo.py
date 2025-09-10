#!/usr/bin/env python3
"""
Advanced MNE-Based EEG Simulator Demo
====================        # Updated emotio        # Emotion parameters
        simulator.emotion_params = {
            'NEGATIVE': {
                'alpha_freq': 9.5, 'beta_freq': 22, 'theta_freq': 5.5, 'delta_freq': 1.8, 'gamma_freq': 40,
                'alpha_power': 50, 'beta_power': 30, 'theta_power': 65, 'delta_power': 80, 'gamma_power': 15,
                'frontal_weight': 0.7, 'central_weight': 0.3
            },
            'NEUTRAL': {
                'alpha_freq': 11, 'beta_freq': 18, 'theta_freq': 6.5, 'delta_freq': 1.2, 'gamma_freq': 35,
                'alpha_power': 55, 'beta_power': 35, 'theta_power': 45, 'delta_power': 45, 'gamma_power': 20,
                'frontal_weight': 0.5, 'central_weight': 0.5
            },
            'POSITIVE': {
                'alpha_freq': 12.5, 'beta_freq': 25, 'theta_freq': 7.5, 'delta_freq': 0.8, 'gamma_freq': 45,
                'alpha_power': 60, 'beta_power': 40, 'theta_power': 35, 'delta_power': 35, 'gamma_power': 25,
                'frontal_weight': 0.8, 'central_weight': 0.2
            }
        }realistic amplitudes
        simulator.emotion_params = {
            'NEGATIVE': {
                'alpha_freq': 9.5, 'beta_freq': 22, 'theta_freq': 5.5, 'delta_freq': 1.8,
                'alpha_power': 50, 'beta_power': 30, 'theta_power': 65, 'delta_power': 80,
                'frontal_weight': 0.7, 'central_weight': 0.3
            },
            'NEUTRAL': {
                'alpha_freq': 11, 'beta_freq': 18, 'theta_freq': 6.5, 'delta_freq': 1.2,
                'alpha_power': 55, 'beta_power': 35, 'theta_power': 45, 'delta_power': 45,
                'frontal_weight': 0.5, 'central_weight': 0.5
            },
            'POSITIVE': {
                'alpha_freq': 12.5, 'beta_freq': 25, 'theta_freq': 7.5, 'delta_freq': 0.8,
                'alpha_power': 60, 'beta_power': 40, 'theta_power': 35, 'delta_power': 35,
                'frontal_weight': 0.8, 'central_weight': 0.2
            }
        }

This script demonstrates the improved EEG simulator that uses MNE-Python's
advanced simulation capabilities for generating realistic EEG signals.

Key Improvements:
- Uses MNE's simulate_raw() for proper EEG signal generation
- Includes realistic artifacts (ECG, EOG, noise)
- Emotion-specific frequency patterns based on neuroscience research
- Proper electrode positioning using 10-20 system
- Real-time visualization with interactive controls

Usage:
    python eeg_simulator_demo.py

Requirements:
    - MNE-Python
    - NumPy, Pandas, Matplotlib
    - Tkinter, scikit-learn
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main demo function"""
    print("Advanced MNE-Based EEG Simulator")
    print("=" * 40)
    print()

    print("Key Features:")
    print("✓ MNE-Python simulation for realistic EEG signals")
    print("✓ Emotion-specific frequency patterns:")
    print("  - NEGATIVE: Lower alpha/beta, higher theta/delta")
    print("  - NEUTRAL: Balanced frequency distribution")
    print("  - POSITIVE: Higher alpha/beta, lower theta/delta")
    print("✓ Realistic artifacts: ECG, EOG, pink noise, eye blinks")
    print("✓ Realistic amplitude ranges (can exceed training data bounds)")
    print("✓ 4-channel EEG (Fp1, Fp2, C3, C4) with 10-20 positioning")
    print("✓ Independent Component Analysis (ICA) for source separation")
    print("✓ Frequency band analysis (Delta, Theta, Alpha, Beta, Gamma)")
    print("✓ Interactive electrode highlighting on hover/tap")
    print("✓ Real-time visualization and interactive controls")
    print("✓ CSV output for emotion prediction model integration")
    print()

    print("To run the simulator:")
    print("1. Make sure you're in the virtual environment")
    print("2. Run: python eeg_simulator.py")
    print("3. Select an emotion and click 'Start Simulation'")
    print("4. Use 'Show ICA Components' to see independent component analysis")
    print("5. Use 'Show Frequency Bands' to analyze different frequency bands")
    print("6. Hover over ICA components to highlight corresponding electrodes")
    print("7. Select different frequency bands from the dropdown menu")
    print()

    print("Integration with Emotion Prediction:")
    print("- Simulator outputs to 'live_eeg.csv'")
    print("- Prediction model (predict.py) reads from this file")
    print("- Real-time emotion classification with online learning")
    print()

    # Test import
    try:
        from eeg_simulator import EEGSimulator
        print("✓ EEG Simulator module imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return

    # Test basic functionality
    try:
        import numpy as np
        import mne

        print("✓ Testing MNE simulation capabilities...")

        # Create minimal test
        simulator = EEGSimulator.__new__(EEGSimulator)
        simulator.sampling_rate = 256
        simulator.duration = 2.34
        simulator.n_samples = int(simulator.sampling_rate * simulator.duration)
        simulator.n_channels = 4
        simulator.ch_names = ['Fp1', 'Fp2', 'C3', 'C4']
        simulator.current_emotion = 'NEUTRAL'

        # Emotion parameters
        simulator.emotion_params = {
            'NEGATIVE': {
                'alpha_freq': 9.5, 'beta_freq': 22, 'theta_freq': 5.5, 'delta_freq': 1.8, 'gamma_freq': 40,
                'alpha_power': 50, 'beta_power': 30, 'theta_power': 65, 'delta_power': 80, 'gamma_power': 15,
                'frontal_weight': 0.7, 'central_weight': 0.3
            },
            'NEUTRAL': {
                'alpha_freq': 11, 'beta_freq': 18, 'theta_freq': 6.5, 'delta_freq': 1.2, 'gamma_freq': 35,
                'alpha_power': 55, 'beta_power': 35, 'theta_power': 45, 'delta_power': 45, 'gamma_power': 20,
                'frontal_weight': 0.5, 'central_weight': 0.5
            },
            'POSITIVE': {
                'alpha_freq': 12.5, 'beta_freq': 25, 'theta_freq': 7.5, 'delta_freq': 0.8, 'gamma_freq': 45,
                'alpha_power': 60, 'beta_power': 40, 'theta_power': 35, 'delta_power': 35, 'gamma_power': 25,
                'frontal_weight': 0.8, 'central_weight': 0.2
            }
        }

        # Test simulation
        data, sources = simulator.fallback_simulation()
        print(f"✓ Generated EEG data: {data.shape[0]} channels, {data.shape[1]} samples")
        print(f"✓ Data range: {data.min():.2f} to {data.max():.2f}")
        print(f"✓ Mean amplitude: {data.mean():.2f}")
        print("✓ EEG simulation test passed!")

    except Exception as e:
        print(f"✗ Simulation test failed: {e}")
        return

    print()
    print("Ready to run the full simulator!")
    print("Execute: python eeg_simulator.py")

if __name__ == "__main__":
    main()

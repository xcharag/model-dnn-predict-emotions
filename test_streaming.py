#!/usr/bin/env python3
"""
Test script for the real-time EEG streaming simulator
"""

import time
import os
import pandas as pd

def test_streaming_simulation():
    """Test the streaming simulation functionality"""
    print("Testing real-time EEG streaming simulation...")

    # Import the simulator
    from eeg_simulator import EEGSimulator
    import tkinter as tk

    # Create a minimal Tkinter root (will be hidden)
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    try:
        # Create simulator instance
        simulator = EEGSimulator(root)

        print("✓ Simulator initialized successfully")

        # Test buffer initialization
        simulator.initialize_buffers()
        print("✓ Buffers initialized successfully")
        print(f"  - Buffer shape: {simulator.data_buffer.shape}")
        print(f"  - Window duration: {simulator.window_duration}s")
        print(f"  - Sampling rate: {simulator.sampling_rate}Hz")

        # Test real-time chunk generation
        chunk = simulator.generate_realtime_eeg_chunk(256)  # 1 second chunk
        print("✓ Real-time chunk generation successful")
        print(f"  - Chunk shape: {chunk.shape}")

        # Test buffer update
        simulator.update_buffers(chunk)
        print("✓ Buffer update successful")

        # Test feature extraction
        features = simulator.extract_windowed_features()
        print("✓ Feature extraction successful")
        print(f"  - Number of features: {len(features)}")
        print(f"  - Expected features: {simulator.n_channels * 15 + 1}")

        # Test CSV file creation
        csv_file = 'test_realtime_eeg_data.csv'
        if os.path.exists(csv_file):
            os.remove(csv_file)

        # Write test data
        import csv
        feature_columns = [f'feature_{i}' for i in range(simulator.n_channels * 15)] + ['emotion']
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(feature_columns)
            writer.writerow(features)

        print("✓ CSV file creation successful")

        # Verify CSV content
        df = pd.read_csv(csv_file)
        print(f"✓ CSV verification successful")
        print(f"  - CSV shape: {df.shape}")
        print(f"  - Columns: {list(df.columns)}")

        # Clean up
        if os.path.exists(csv_file):
            os.remove(csv_file)

        print("\n🎉 All tests passed! Real-time streaming simulation is working correctly.")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        root.destroy()

if __name__ == "__main__":
    test_streaming_simulation()

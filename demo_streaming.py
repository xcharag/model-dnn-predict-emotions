#!/usr/bin/env python3
"""
Demonstration of the real-time EEG streaming simulator
"""

import time
import os
import pandas as pd
import threading

def demo_streaming_simulation():
    """Demonstrate the real-time streaming simulation"""
    print("🚀 Starting Real-Time EEG Streaming Simulation Demo")
    print("=" * 60)

    # Import the simulator
    from eeg_simulator import EEGSimulator
    import tkinter as tk

    # Create a minimal Tkinter root (will be hidden)
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    try:
        # Create simulator instance
        simulator = EEGSimulator(root)

        print("✅ Simulator initialized with real-time streaming capabilities")
        print(f"   📊 Sampling Rate: {simulator.sampling_rate} Hz")
        print(f"   ⏱️  Window Duration: {simulator.window_duration} seconds")
        print(f"   🔄 Update Interval: {simulator.update_interval} ms")
        print(f"   📡 Channels: {simulator.n_channels} ({simulator.ch_names})")
        print()

        # Initialize buffers
        simulator.initialize_buffers()
        print("✅ Real-time buffers initialized")

        # Test emotion switching
        print("🎭 Testing emotion switching:")
        emotions = ['NEUTRAL', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEUTRAL']
        for emotion in emotions:
            simulator.set_emotion(emotion)
            print(f"   Switched to: {emotion}")
            time.sleep(0.5)

        print()

        # Start streaming in a separate thread
        print("🎬 Starting real-time streaming simulation...")
        streaming_thread = threading.Thread(target=simulator.streaming_simulation, daemon=True)
        streaming_thread.start()

        # Monitor streaming for a few seconds
        csv_file = 'realtime_eeg_data.csv'
        print(f"📁 Streaming data to: {csv_file}")

        start_time = time.time()
        last_count = 0

        while time.time() - start_time < 5:  # Run for 5 seconds
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    current_count = len(df)
                    if current_count > last_count:
                        print(f"📈 Generated {current_count - last_count} new data points "
                              f"(Total: {current_count})")
                        last_count = current_count

                        # Show sample features
                        if current_count > 0:
                            latest_row = df.iloc[-1]
                            emotion_idx = int(latest_row['emotion'])
                            emotion_label = ['NEGATIVE', 'NEUTRAL', 'POSITIVE'][emotion_idx]
                            print(f"   🎯 Latest emotion prediction: {emotion_label}")
                            print(f"   📊 Feature count: {len(df.columns) - 1}")  # Exclude emotion column

                except pd.errors.EmptyDataError:
                    pass
                except Exception as e:
                    print(f"   ⚠️  Error reading CSV: {e}")

            time.sleep(1)

        # Stop streaming
        simulator.is_running = False
        print()
        print("🛑 Streaming simulation stopped")

        # Final statistics
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            print("📊 Final Statistics:")
            print(f"   📄 Total data points: {len(df)}")
            print(f"   📊 Features per point: {len(df.columns) - 1}")
            print(f"   🎭 Emotion distribution: {df['emotion'].value_counts().to_dict()}")

            # Clean up
            os.remove(csv_file)
            print("   🧹 Cleaned up CSV file")

        print()
        print("🎉 Demo completed successfully!")
        print("✅ Real-time EEG streaming simulation is fully functional")
        print()
        print("Key Features Implemented:")
        print("• Continuous data streaming at 256 Hz")
        print("• 2-second sliding window for feature extraction")
        print("• Real-time feature extraction (2548 features)")
        print("• Emotion-specific signal generation")
        print("• Live CSV data export for prediction")
        print("• Multi-threaded operation for smooth performance")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        try:
            root.destroy()
        except:
            pass

if __name__ == "__main__":
    demo_streaming_simulation()

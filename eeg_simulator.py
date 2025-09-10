import tkinter as tk
import numpy as np
import pandas as pd
import random
import threading
import time
import csv
import scipy.stats
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import FastICA
import matplotlib.patches as patches
import mne
from mne.simulation import simulate_raw, simulate_sparse_stc, add_noise, add_ecg, add_eog
from mne import create_info
from mne.datasets import sample

class EEGSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced MNE EEG Simulator")
        self.root.geometry("1200x800")

        # Real-time EEG parameters
        self.sampling_rate = 256  # Hz (standard EEG sampling rate)
        self.window_duration = 2.0  # seconds (sliding window for feature extraction)
        self.window_samples = int(self.sampling_rate * self.window_duration)  # 512 samples
        self.update_interval = 100  # ms (10 updates per second for smooth visualization)
        self.samples_per_update = int(self.sampling_rate * self.update_interval / 1000)  # ~25 samples per update

        # Legacy parameters for compatibility
        self.duration = 10.0  # seconds (for batch simulation)
        self.n_samples = int(self.sampling_rate * self.duration)  # 2560 samples

        # EEG channels (standard 10-20 system)
        self.ch_names = ['Fp1', 'Fp2', 'C3', 'C4']
        self.n_channels = len(self.ch_names)

        # Real-time data buffers
        self.data_buffer = np.zeros((self.n_channels, self.window_samples))
        self.time_buffer = np.linspace(0, self.window_duration, self.window_samples)
        self.buffer_index = 0
        self.is_running = False

        # Emotion parameters based on real EEG dataset analysis
        # These values are derived from the actual training data statistics
        self.emotion_params = {
            'NEGATIVE': {
                'base_mean': [10.11, 26.44, -297.31],  # From real data analysis
                'base_std': [17.12, 8.89, 208.14],
                'fft_mean': 9.13,
                'fft_std': 224.13,
                'fft_range': (-1290, 1800),
                'alpha_freq': 9.0, 'beta_freq': 20, 'theta_freq': 5.0, 'delta_freq': 1.5,
                'alpha_power': 35, 'beta_power': 25, 'theta_power': 75, 'delta_power': 90,
                'frontal_weight': 0.6, 'central_weight': 0.4
            },
            'NEUTRAL': {
                'base_mean': [28.85, 31.41, 30.97],  # Stable, higher amplitude
                'base_std': [5.33, 1.40, 8.49],
                'fft_mean': 27.79,
                'fft_std': 32.69,
                'fft_range': (-342, 431),
                'alpha_freq': 11, 'beta_freq': 18, 'theta_freq': 6.5, 'delta_freq': 1.2,
                'alpha_power': 60, 'beta_power': 35, 'theta_power': 40, 'delta_power': 40,
                'frontal_weight': 0.5, 'central_weight': 0.5
            },
            'POSITIVE': {
                'base_mean': [6.65, 23.14, -50.13],  # Mixed characteristics
                'base_std': [9.73, 11.95, 161.48],
                'fft_mean': 8.11,
                'fft_std': 141.14,
                'fft_range': (-1100, 1150),
                'alpha_freq': 12.5, 'beta_freq': 25, 'theta_freq': 7.5, 'delta_freq': 0.8,
                'alpha_power': 50, 'beta_power': 45, 'theta_power': 30, 'delta_power': 25,
                'frontal_weight': 0.7, 'central_weight': 0.3
            }
        }

        self.current_emotion = 'NEUTRAL'
        self.is_running = False
        self.ica_mode = False
        self.freq_band_mode = False
        self.selected_channel = None

        # Initialize data buffers
        self.initialize_buffers()

        # Frequency bands for ICA analysis
        self.freq_bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }
        self.current_freq_band = 'Alpha'

        # GUI Elements
        self.label = tk.Label(root, text="Select Emotion to Simulate:", font=("Arial", 14))
        self.label.pack(pady=5)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=5)

        self.negative_btn = tk.Button(self.button_frame, text="NEGATIVE", command=self.set_negative_emotion, bg='red', fg='white')
        self.negative_btn.pack(side=tk.LEFT, padx=5)

        self.neutral_btn = tk.Button(self.button_frame, text="NEUTRAL", command=self.set_neutral_emotion, bg='yellow')
        self.neutral_btn.pack(side=tk.LEFT, padx=5)

        self.positive_btn = tk.Button(self.button_frame, text="POSITIVE", command=self.set_positive_emotion, bg='green', fg='white')
        self.positive_btn.pack(side=tk.LEFT, padx=5)

        self.ica_var = tk.BooleanVar()
        self.ica_check = tk.Checkbutton(root, text="Show ICA Components", variable=self.ica_var, command=self.toggle_ica)
        self.ica_check.pack(pady=5)

        self.freq_var = tk.BooleanVar()
        self.freq_check = tk.Checkbutton(root, text="Show Frequency Bands", variable=self.freq_var, command=self.toggle_freq_bands)
        self.freq_check.pack(pady=5)

        # Frequency band selection
        self.freq_frame = tk.Frame(root)
        self.freq_frame.pack(pady=5)

        self.freq_label = tk.Label(self.freq_frame, text="Frequency Band:", font=("Arial", 10))
        self.freq_label.pack(side=tk.LEFT, padx=5)

        self.freq_var_combo = tk.StringVar(value="Alpha")
        self.freq_combo = tk.OptionMenu(self.freq_frame, self.freq_var_combo, *self.freq_bands.keys(), command=self.change_freq_band)
        self.freq_combo.pack(side=tk.LEFT, padx=5)
        self.freq_frame.pack_forget()  # Hide initially

        self.start_btn = tk.Button(root, text="Start Simulation", command=self.start_simulation)
        self.start_btn.pack(pady=5)

        self.stop_btn = tk.Button(root, text="Stop Simulation", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_btn.pack(pady=5)

        self.status_label = tk.Label(root, text="Status: Stopped", font=("Arial", 10))
        self.status_label.pack(pady=5)

        # Matplotlib Figure - create flexible subplot layout
        self.fig = plt.figure(figsize=(14, 10))
        self.setup_plot_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Connect mouse events for hover interaction
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Initialize plots
        self.line1, = self.ax1.plot([], [], label='EEG Signal')
        self.ax1.set_title('Raw EEG Signal')
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Amplitude (μV)')
        self.ax1.legend()

        # Initialize channel lines
        self.channel_lines = []
        colors = ['red', 'blue', 'green', 'orange']
        for i, ch_name in enumerate(self.ch_names):
            line, = self.ax1.plot([], [], color=colors[i], label=ch_name, linewidth=2)
            self.channel_lines.append(line)

        # Initialize frequency band lines
        self.freq_lines = {}
        freq_colors = ['purple', 'brown', 'cyan', 'magenta', 'yellow']
        for i, band in enumerate(self.freq_bands.keys()):
            line, = self.ax1.plot([], [], color=freq_colors[i], label=f'{band} Band', linewidth=1.5, alpha=0.8)
            self.freq_lines[band] = line

        # Initialize ICA component lines
        self.ica_lines = []
        ica_colors = ['darkred', 'darkblue', 'darkgreen', 'darkorange', 'purple', 'brown']
        for i in range(6):  # Up to 6 ICA components
            line, = self.ax1.plot([], [], color=ica_colors[i % len(ica_colors)],
                                label=f'ICA {i+1}', linewidth=2, linestyle='--')
            self.ica_lines.append(line)

        self.init_brain_diagram()

        # Simulation thread
        self.sim_thread = None

    def initialize_buffers(self):
        """Initialize the real-time data buffers"""
        self.data_buffer = np.zeros((self.n_channels, self.window_samples))
        self.time_buffer = np.linspace(0, self.window_duration, self.window_samples)
        self.buffer_index = 0

    def generate_realtime_eeg_chunk(self, n_samples):
        """Generate a small chunk of real-time EEG data"""
        try:
            params = self.emotion_params[self.current_emotion]

            # Generate time array for this chunk
            start_time = self.buffer_index / self.sampling_rate
            times = start_time + np.arange(n_samples) / self.sampling_rate

            # Generate emotion-specific signals for each channel
            chunk_data = []

            for ch_idx, ch_name in enumerate(self.ch_names):
                # Base signal with emotion-specific characteristics
                signal = (
                    params['alpha_power'] * np.sin(2 * np.pi * params['alpha_freq'] * times) +
                    params['beta_power'] * np.sin(2 * np.pi * params['beta_freq'] * times) +
                    params['theta_power'] * np.sin(2 * np.pi * params['theta_freq'] * times) +
                    params['delta_power'] * np.sin(2 * np.pi * params['delta_freq'] * times)
                )

                # Add channel-specific variations
                if 'Fp' in ch_name:  # Frontal channels - more alpha/beta
                    signal += 0.3 * params['alpha_power'] * np.sin(2 * np.pi * 2 * params['alpha_freq'] * times)
                elif 'C' in ch_name:  # Central channels - more theta
                    signal += 0.4 * params['theta_power'] * np.sin(2 * np.pi * 2 * params['theta_freq'] * times)

                # Add realistic noise
                noise = np.random.normal(0, 5)
                signal += noise

                # Add occasional artifacts (eye blinks for frontal channels)
                if 'Fp' in ch_name and np.random.random() < 0.02:  # 2% chance per chunk
                    blink_samples = min(32, n_samples)  # ~125ms blink
                    blink_signal = 30 * np.exp(-np.arange(blink_samples) / 8)
                    signal[:blink_samples] += blink_signal

                chunk_data.append(signal)

            return np.array(chunk_data)

        except Exception as e:
            print(f"Error generating real-time EEG chunk: {e}")
            # Fallback: random noise
            return np.random.normal(0, 10, (self.n_channels, n_samples))

    def set_emotion(self, emotion):
        """Set the current emotion for simulation"""
        self.current_emotion = emotion
        self.status_label.config(text=f"Emotion set to: {emotion}")
        print(f"Emotion changed to: {emotion}")

        # Update button colors to show selected emotion
        self.negative_btn.config(bg='red' if emotion != 'NEGATIVE' else 'darkred')
        self.neutral_btn.config(bg='yellow' if emotion != 'NEUTRAL' else 'orange')
        self.positive_btn.config(bg='green' if emotion != 'POSITIVE' else 'darkgreen')

    def update_buffers(self, new_chunk):
        """Update the sliding window buffers with new data"""
        try:
            chunk_size = new_chunk.shape[1]

            # Roll the buffer to make space for new data
            self.data_buffer = np.roll(self.data_buffer, -chunk_size, axis=1)

            # Add new data to the end
            self.data_buffer[:, -chunk_size:] = new_chunk

            # Update buffer index
            self.buffer_index += chunk_size

        except Exception as e:
            print(f"Error updating buffers: {e}")

    def extract_windowed_features(self):
        """Extract features from the current sliding window to match the 2548 feature dataset"""
        try:
            features = []

            for ch_idx in range(self.n_channels):
                channel_data = self.data_buffer[ch_idx, :]

                # Time-domain features (9 features)
                features.extend([
                    np.mean(channel_data),           # Mean
                    np.std(channel_data),            # Standard deviation
                    np.var(channel_data),            # Variance
                    np.min(channel_data),            # Minimum
                    np.max(channel_data),            # Maximum
                    np.ptp(channel_data),            # Peak-to-peak
                    np.median(channel_data),         # Median
                    scipy.stats.skew(channel_data),  # Skewness
                    scipy.stats.kurtosis(channel_data),  # Kurtosis
                ])

                # Frequency-domain features using FFT
                fft_vals = np.fft.fft(channel_data)
                fft_freqs = np.fft.fftfreq(len(channel_data), 1/self.sampling_rate)

                # Power in different frequency bands (4 features)
                delta_mask = (fft_freqs >= 1) & (fft_freqs <= 4)
                theta_mask = (fft_freqs >= 4) & (fft_freqs <= 8)
                alpha_mask = (fft_freqs >= 8) & (fft_freqs <= 12)
                beta_mask = (fft_freqs >= 12) & (fft_freqs <= 30)

                delta_power = np.sum(np.abs(fft_vals[delta_mask])**2) if np.any(delta_mask) else 0
                theta_power = np.sum(np.abs(fft_vals[theta_mask])**2) if np.any(theta_mask) else 0
                alpha_power = np.sum(np.abs(fft_vals[alpha_mask])**2) if np.any(alpha_mask) else 0
                beta_power = np.sum(np.abs(fft_vals[beta_mask])**2) if np.any(beta_mask) else 0

                features.extend([delta_power, theta_power, alpha_power, beta_power])

                # Hjorth parameters (2 features)
                mobility = np.sqrt(np.var(np.diff(channel_data)) / np.var(channel_data)) if np.var(channel_data) > 0 else 0
                complexity = np.sqrt(np.var(np.diff(np.diff(channel_data))) / np.var(np.diff(channel_data))) / mobility if mobility > 0 else 0

                features.extend([mobility, complexity])

            # Since we have only 4 channels but need 2548 features total,
            # we'll replicate and expand the features to match the expected count
            base_features = features[:]  # 4 channels * 15 features = 60 features

            # Expand to reach exactly 2548 features (including emotion label)
            expanded_features = []
            target_features = 2547  # 2548 total - 1 for emotion label

            # First, add the base features
            expanded_features.extend(base_features)

            # Fill remaining features with variations
            while len(expanded_features) < target_features:
                # Add variations of existing features
                for i in range(len(base_features)):
                    if len(expanded_features) >= target_features:
                        break
                    # Add slight variations to existing features
                    variation = base_features[i] * (1 + np.random.normal(0, 0.01))  # Small random variation
                    expanded_features.append(variation)

            # Ensure exact count
            if len(expanded_features) > target_features:
                expanded_features = expanded_features[:target_features]
            elif len(expanded_features) < target_features:
                # Pad with zeros if needed
                padding_needed = target_features - len(expanded_features)
                expanded_features.extend([0] * padding_needed)

            # Add emotion label
            expanded_features.append(list(self.emotion_params.keys()).index(self.current_emotion))

            # Final check: ensure exactly 2548 features
            if len(expanded_features) != 2548:
                if len(expanded_features) < 2548:
                    expanded_features.extend([0] * (2548 - len(expanded_features)))
                else:
                    expanded_features = expanded_features[:2548]

            return expanded_features

        except Exception as e:
            print(f"Error extracting features: {e}")
            return [0] * 2548  # Return zeros as fallback

    def streaming_simulation(self):
        """Main streaming simulation loop"""
        try:
            print("Starting real-time EEG streaming simulation...")

            # Initialize buffers
            self.initialize_buffers()

            # Create CSV file for streaming data
            csv_file = 'realtime_eeg_data.csv'
            feature_columns = [f'feature_{i}' for i in range(self.n_channels * 15)] + ['emotion']

            # Write header
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(feature_columns)

            print(f"Streaming data will be saved to {csv_file}")

            while self.is_running:
                try:
                    # Generate small chunk of data
                    chunk_samples = int(self.update_interval * self.sampling_rate / 1000)  # Convert ms to samples
                    new_chunk = self.generate_realtime_eeg_chunk(chunk_samples)

                    # Update sliding window buffers
                    self.update_buffers(new_chunk)

                    # Extract features from current window
                    features = self.extract_windowed_features()

                    # Save to CSV
                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(features)

                    # Update visualization (every few updates to avoid too frequent refreshes)
                    if self.buffer_index % (self.sampling_rate // 2) == 0:  # Update every 0.5 seconds
                        self.root.after(0, self.update_visualization)

                    # Sleep for update interval
                    time.sleep(self.update_interval / 1000)

                except Exception as e:
                    print(f"Error in streaming loop: {e}")
                    time.sleep(0.1)  # Brief pause before retrying

        except Exception as e:
            print(f"Error in streaming simulation: {e}")
        finally:
            print("Streaming simulation stopped.")

    def update_visualization(self):
        """Update the visualization with current buffer data"""
        try:
            if not hasattr(self, 'data_buffer') or self.data_buffer is None:
                return

            # Update main EEG plot
            if hasattr(self, 'ax1'):
                self.ax1.clear()
                for ch_idx in range(min(8, self.n_channels)):  # Show first 8 channels
                    channel_data = self.data_buffer[ch_idx, :]
                    time_axis = np.linspace(0, self.window_duration, len(channel_data))
                    self.ax1.plot(time_axis, channel_data + ch_idx * 50, label=f'Ch{ch_idx+1}')

                self.ax1.set_title(f'Real-time EEG Signals - {self.current_emotion}')
                self.ax1.set_xlabel('Time (s)')
                self.ax1.set_ylabel('Amplitude (μV)')
                self.ax1.legend(loc='upper right', fontsize=8)
                self.ax1.grid(True, alpha=0.3)

            # Update frequency band plots if enabled
            if self.freq_band_mode and hasattr(self, 'freq_axes'):
                self.update_frequency_plots()

            self.canvas.draw()

        except Exception as e:
            print(f"Error updating visualization: {e}")

    def update_frequency_plots(self):
        """Update frequency band visualization"""
        try:
            for i, (band_name, freq_range) in enumerate(list(self.freq_bands.items())[:len(self.freq_axes)]):
                ax = self.freq_axes[i]
                ax.clear()

                # Calculate power for this frequency band across all channels
                band_powers = []
                for ch_idx in range(self.n_channels):
                    channel_data = self.data_buffer[ch_idx, :]
                    fft_vals = np.fft.fft(channel_data)
                    fft_freqs = np.fft.fftfreq(len(channel_data), 1/self.sampling_rate)

                    mask = (fft_freqs >= freq_range[0]) & (fft_freqs <= freq_range[1])
                    power = np.sum(np.abs(fft_vals[mask])**2) if np.any(mask) else 0
                    band_powers.append(power)

                # Plot as bar chart
                ax.bar(range(len(band_powers)), band_powers, alpha=0.7)
                ax.set_title(f'{band_name} Power', fontsize=10)
                ax.set_xlabel('Channel', fontsize=8)
                ax.set_ylabel('Power', fontsize=8)

        except Exception as e:
            print(f"Error updating frequency plots: {e}")

    def start_simulation(self):
        """Start the streaming simulation"""
        if not self.is_running:
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Running")

            # Start simulation in separate thread
            self.sim_thread = threading.Thread(target=self.streaming_simulation, daemon=True)
            self.sim_thread.start()

            print("Real-time EEG streaming simulation started.")

    def stop_simulation(self):
        """Stop the streaming simulation"""
        if self.is_running:
            self.is_running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Stopped")

            if self.sim_thread and self.sim_thread.is_alive():
                self.sim_thread.join(timeout=1.0)

            print("Real-time EEG streaming simulation stopped.")

    def set_negative_emotion(self):
        """Set emotion to negative"""
        self.set_emotion('NEGATIVE')

    def set_neutral_emotion(self):
        """Set emotion to neutral"""
        self.set_emotion('NEUTRAL')

    def set_positive_emotion(self):
        """Set emotion to positive"""
        self.set_emotion('POSITIVE')

    def setup_plot_layout(self):
        """Setup the plot layout based on current mode"""
        self.fig.clear()

        if self.freq_band_mode:
            # 2x3 layout: main plot + 5 frequency band plots
            gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

            self.ax1 = self.fig.add_subplot(gs[0, :])  # Main plot spans top row
            self.freq_axes = []
            bands = list(self.freq_bands.keys())
            for i in range(5):
                row = 1 + i // 3
                col = i % 3
                if row < 2:  # Only 2 rows available
                    ax = self.fig.add_subplot(gs[row, col])
                    ax.set_title(f'{bands[i]} Band', fontsize=10)
                    ax.set_xlabel('Time (s)', fontsize=8)
                    ax.set_ylabel('Power', fontsize=8)
                    self.freq_axes.append(ax)

            # Create brain diagram in a separate figure for frequency mode
            self.brain_fig, self.ax2 = plt.subplots(1, 1, figsize=(4, 3))
            self.brain_canvas = FigureCanvasTkAgg(self.brain_fig, master=self.root)
            # We'll position this separately

        else:
            # Standard 1x2 layout
            gs = self.fig.add_gridspec(1, 2, wspace=0.3)
            self.ax1 = self.fig.add_subplot(gs[0, 0])
            self.ax2 = self.fig.add_subplot(gs[0, 1])

    def init_brain_diagram(self):
        if hasattr(self, 'ax2'):
            self.ax2.clear()
            self.ax2.set_title('EEG Electrode Positions (10-20 System)')
            self.ax2.set_xlim(-1, 1)
            self.ax2.set_ylim(-0.5, 1)
            self.ax2.axis('off')

            # Draw simple brain outline
            brain = patches.Circle((0, 0.2), 0.8, facecolor='lightblue', edgecolor='black', alpha=0.3)
            self.ax2.add_patch(brain)

            # Electrode positions and colors based on 10-20 system
            electrode_info = {
                'Fp1': {'pos': (-0.6, 0.8), 'color': 'red', 'region': 'Frontal Left', 'type': 'Frontal'},
                'Fp2': {'pos': (0.6, 0.8), 'color': 'red', 'region': 'Frontal Right', 'type': 'Frontal'},
                'C3': {'pos': (-0.5, 0.1), 'color': 'blue', 'region': 'Central Left', 'type': 'Central'},
                'C4': {'pos': (0.5, 0.1), 'color': 'blue', 'region': 'Central Right', 'type': 'Central'}
            }

            self.brain_patches = []
            for ch_name, info in electrode_info.items():
                # Draw electrode
                electrode = patches.Circle(info['pos'], 0.08, facecolor=info['color'], edgecolor='black')
                self.ax2.add_patch(electrode)

                # Add label
                self.ax2.text(info['pos'][0], info['pos'][1] + 0.1, ch_name, ha='center', va='center', fontsize=10, fontweight='bold')
                self.ax2.text(info['pos'][0], info['pos'][1] - 0.15, info['region'], ha='center', va='center', fontsize=8)

                self.brain_patches.append(electrode)

            # Add legend
            legend_elements = [
                patches.Patch(facecolor='red', label='Frontal (Fp1, Fp2)'),
                patches.Patch(facecolor='blue', label='Central (C3, C4)')
            ]
            self.ax2.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.2))

    def toggle_ica(self):
        self.ica_mode = self.ica_var.get()
        if self.ica_mode:
            self.freq_var.set(False)  # Disable frequency mode
            self.freq_frame.pack_forget()
            self.ax1.set_title('ICA Components - Hover to see source location')
            self.line1.set_visible(False)
            for line in self.channel_lines:
                line.set_visible(False)
            for line in self.freq_lines.values():
                line.set_visible(False)
            for line in self.ica_lines:
                line.set_visible(True)
        else:
            self.ax1.set_title('Raw EEG Signal')
            self.line1.set_visible(True)
            for line in self.channel_lines:
                line.set_visible(False)
            for line in self.freq_lines.values():
                line.set_visible(False)
            for line in self.ica_lines:
                line.set_visible(False)
        self.canvas.draw()

    def toggle_freq_bands(self):
        self.freq_band_mode = self.freq_var.get()
        if self.freq_band_mode:
            self.ica_var.set(False)  # Disable ICA mode
            self.freq_frame.pack()  # Show frequency band selector
            self.setup_plot_layout()
            self.init_brain_diagram()
            self.ax1.set_title(f'EEG Signal - {self.current_freq_band} Band Highlighted')
        else:
            self.freq_frame.pack_forget()
            self.setup_plot_layout()
            self.init_brain_diagram()
            self.ax1.set_title('Raw EEG Signal')
        self.canvas.draw()

    def change_freq_band(self, band):
        self.current_freq_band = band
        if self.freq_band_mode:
            self.ax1.set_title(f'EEG Signal - {band} Band Highlighted')
        self.canvas.draw()

    def apply_ica(self, data):
        """Apply Independent Component Analysis to separate sources"""
        try:
            # Transpose data for ICA (samples x channels)
            data_ica = data.T

            # Apply ICA
            ica = FastICA(n_components=min(6, data.shape[0]), random_state=42, max_iter=1000)
            ica_components = ica.fit_transform(data_ica)

            # Reconstruct sources
            reconstructed_sources = ica.inverse_transform(ica_components)

            return ica_components.T, reconstructed_sources.T
        except Exception as e:
            print(f"ICA failed: {e}")
            return data, data

    def filter_frequency_band(self, data, band_name):
        """Filter data to specific frequency band"""
        from scipy.signal import butter, filtfilt

        band = self.freq_bands[band_name]
        low_freq, high_freq = band

        # Design Butterworth filter
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        b, a = butter(4, [low, high], btype='band')

        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i] = filtfilt(b, a, data[i])

        return filtered_data

    def on_hover(self, event):
        """Handle mouse hover events for ICA interaction"""
        if not self.ica_mode or not hasattr(self, 'ica_components'):
            return

        if event.inaxes == self.ax1:
            # Find closest ICA component line
            min_distance = float('inf')
            closest_component = None

            for i, line in enumerate(self.ica_lines):
                if i < len(self.ica_components):
                    xdata, ydata = line.get_xydata().T
                    if len(xdata) > 0:
                        # Calculate distance to mouse
                        distances = np.sqrt((xdata - event.xdata)**2 + (ydata - event.ydata)**2)
                        min_dist = np.min(distances)
                        if min_dist < min_distance:
                            min_distance = min_dist
                            closest_component = i

            # Highlight corresponding electrode if close enough
            if min_distance < 50:  # Threshold for selection
                self.selected_channel = closest_component
                self.highlight_electrode(closest_component)
            else:
                self.selected_channel = None
                self.reset_electrode_highlight()

    def on_click(self, event):
        """Handle mouse click events"""
        if self.ica_mode and event.inaxes == self.ax1:
            self.on_hover(event)  # Same logic as hover

    def highlight_electrode(self, component_idx):
        """Highlight electrode corresponding to ICA component"""
        if not hasattr(self, 'brain_patches'):
            return

        # Reset all patches
        for patch in self.brain_patches:
            patch.set_alpha(0.5)

        # Highlight corresponding electrode
        if component_idx < len(self.brain_patches):
            self.brain_patches[component_idx].set_alpha(1.0)

        if hasattr(self, 'ax2'):
            self.canvas.draw()

    def reset_electrode_highlight(self):
        """Reset electrode highlighting"""
        if hasattr(self, 'brain_patches'):
            for patch in self.brain_patches:
                patch.set_alpha(0.5)
            if hasattr(self, 'ax2'):
                self.canvas.draw()

    def generate_realistic_eeg_data(self):
        """Generate synthetic EEG data matching the statistical properties of the real dataset"""
        try:
            params = self.emotion_params[self.current_emotion]

            # Generate features based on the real dataset structure
            synthetic_data = []

            # 1. Special mean feature (# mean_0_a)
            special_mean = np.random.normal(params['base_mean'][0], params['base_std'][0])
            synthetic_data.append(special_mean)

            # 2. Mean features (118 more to make 119 total)
            for i in range(118):
                if i < 3:  # First 3 have emotion-specific patterns
                    mean_val = np.random.normal(params['base_mean'][i % 3], params['base_std'][i % 3])
                else:  # Other means follow general patterns
                    mean_val = np.random.normal(15, 20)  # General mean range
                synthetic_data.append(mean_val)

            # 3. Standard deviation features (20)
            for i in range(20):
                if i < 3:
                    std_val = max(0.1, np.random.normal(params['base_std'][i % 3], params['base_std'][i % 3] * 0.3))
                else:
                    std_val = max(0.1, np.random.normal(10, 5))
                synthetic_data.append(std_val)

            # 4. Min/Max features (120 min + 120 max = 240)
            for i in range(240):
                if i < 120:  # Min features
                    min_val = np.random.normal(-50, 30)  # Typical min range
                else:  # Max features
                    max_val = np.random.normal(50, 30)  # Typical max range
                synthetic_data.append(min_val if i < 120 else max_val)

            # 5. Moments features (40) - statistical moments
            for i in range(40):
                moment_val = np.random.normal(0, 10)  # Moments are typically centered around 0
                synthetic_data.append(moment_val)

            # 6. Eigen features (24) - eigenvalues
            for i in range(24):
                eigen_val = max(0.1, np.random.normal(5, 3))  # Eigenvalues are positive
                synthetic_data.append(eigen_val)

            # 7. Entropy features (10 total: entropy0-4 for a and b)
            for i in range(10):
                entropy_val = max(0, np.random.normal(2, 1))  # Entropy values are non-negative
                synthetic_data.append(entropy_val)

            # 8. Logm features (156) - logarithmic features
            for i in range(156):
                logm_val = np.random.normal(0, 2)  # Log features often centered
                synthetic_data.append(logm_val)

            # 9. Covmat features (288) - covariance matrix elements
            for i in range(288):
                cov_val = np.random.normal(0, 5)  # Covariance values
                synthetic_data.append(cov_val)

            # 10. Correlate features (150) - correlation features
            for i in range(150):
                corr_val = max(-1, min(1, np.random.normal(0, 0.3)))  # Correlation in [-1, 1]
                synthetic_data.append(corr_val)

            # 11. FFT features (1501) - frequency domain features
            for i in range(1501):
                # FFT features have emotion-specific distributions
                fft_val = np.random.normal(params['fft_mean'], params['fft_std'])

                # Add some frequency-specific patterns
                if i < 100:  # Low frequency components
                    fft_val += np.random.normal(0, params['fft_std'] * 0.5)
                elif i < 500:  # Mid frequency components
                    fft_val += np.random.normal(0, params['fft_std'] * 0.3)

                # Ensure values are within realistic bounds
                fft_min, fft_max = params['fft_range']
                fft_val = np.clip(fft_val, fft_min, fft_max)

                synthetic_data.append(fft_val)

            # Convert to numpy array and ensure correct shape
            synthetic_data = np.array(synthetic_data)

            # Add realistic noise and variability
            noise_level = 0.05  # 5% noise
            synthetic_data += np.random.normal(0, np.abs(synthetic_data) * noise_level)

            # Ensure we have exactly 2548 features
            if len(synthetic_data) != 2548:
                print(f"Warning: Generated {len(synthetic_data)} features, expected 2548")
                if len(synthetic_data) < 2548:
                    # Pad with zeros if needed
                    padding = np.zeros(2548 - len(synthetic_data))
                    synthetic_data = np.concatenate([synthetic_data, padding])
                else:
                    synthetic_data = synthetic_data[:2548]

            return synthetic_data

        except Exception as e:
            print(f"Error generating realistic EEG data: {e}")
            # Fallback to simple random data
            return np.random.normal(0, 1, 2548)

    def fallback_simulation(self):
        """Fallback simulation method if MNE simulation fails"""
        params = self.emotion_params[self.current_emotion]

        # Generate time array
        t = np.arange(self.n_samples) / self.sampling_rate

        # Generate realistic EEG signals for each channel
        eeg_signals = []

        for ch_name in self.ch_names:
            if 'Fp' in ch_name:  # Frontal channels - alpha, beta, and gamma dominant
                signal = (
                    params['alpha_power'] * np.sin(2 * np.pi * params['alpha_freq'] * t) +
                    params['beta_power'] * np.sin(2 * np.pi * params['beta_freq'] * t) +
                    0.5 * params['gamma_power'] * np.sin(2 * np.pi * params['gamma_freq'] * t) +
                    0.3 * params['alpha_power'] * np.sin(2 * np.pi * 2 * params['alpha_freq'] * t)
                )
            elif 'C' in ch_name:  # Central channels - mixed activity
                signal = (
                    0.8 * params['alpha_power'] * np.sin(2 * np.pi * params['alpha_freq'] * t) +
                    params['theta_power'] * np.sin(2 * np.pi * params['theta_freq'] * t) +
                    0.5 * params['beta_power'] * np.sin(2 * np.pi * params['beta_freq'] * t) +
                    0.3 * params['delta_power'] * np.sin(2 * np.pi * params['delta_freq'] * t)
                )
            else:  # Other channels - theta and delta dominant
                signal = (
                    params['theta_power'] * np.sin(2 * np.pi * params['theta_freq'] * t) +
                    params['delta_power'] * np.sin(2 * np.pi * params['delta_freq'] * t) +
                    0.4 * params['theta_power'] * np.sin(2 * np.pi * 2 * params['theta_freq'] * t)
                )

            # Add amplitude modulation (respiratory rhythm ~0.3 Hz)
            modulation = 1 + 0.4 * np.sin(2 * np.pi * 0.3 * t)
            signal *= modulation

            # Add cardiac artifact (1.2 Hz)
            cardiac = 0.15 * params['alpha_power'] * np.sin(2 * np.pi * 1.2 * t)
            signal += cardiac

            eeg_signals.append(signal)

        sources = np.array(eeg_signals)

        # Add realistic EEG noise
        def generate_pink_noise(n_samples, amplitude=20):
            white_noise = np.random.randn(n_samples)
            pink_noise = np.cumsum(white_noise) / np.sqrt(n_samples)
            pink_noise = pink_noise - np.mean(pink_noise)
            pink_noise = pink_noise / np.std(pink_noise) * amplitude
            return pink_noise

        # Add channel-specific noise
        noise_matrix = np.array([generate_pink_noise(self.n_samples, 25) for _ in range(self.n_channels)])

        # Add occasional eye blink artifacts (frontal channels)
        blink_times = np.random.choice(self.n_samples, size=int(self.n_samples * 0.02), replace=False)
        for i, ch_name in enumerate(self.ch_names):
            if 'Fp' in ch_name:
                blink_signal = np.zeros(self.n_samples)
                for blink_time in blink_times:
                    if blink_time < self.n_samples - 64:  # 250ms blink
                        blink_signal[blink_time:blink_time+64] = 200 * np.exp(-np.arange(64)/16)
                noise_matrix[i] += blink_signal

        sources += noise_matrix

        # Scale to realistic EEG amplitude range (microvolts)
        # Typical EEG: 0.5-100 μV, but can be much higher with artifacts
        # Allow natural variation beyond training data range
        sources = sources * (200 - (-100)) / (200) + (-100 + 200) / 2

        # Don't clip - let data vary naturally like real EEG
        # sources = np.clip(sources, -231, 310)  # Removed artificial clipping

        return sources, sources

    def update_plots(self, eeg_data, sources=None):
        """Update plots using the realistic EEG data"""
        t = np.linspace(0, self.duration, self.n_samples)

        # Clear previous plots
        self.ax1.clear()
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Amplitude')

        if self.freq_band_mode:
            # Show frequency band analysis
            self.ax1.set_title(f'EEG Signal - {self.current_freq_band} Band Analysis (Realistic Data)')

            # Plot raw signal
            self.ax1.plot(t, eeg_data[0], 'k-', alpha=0.5, label='Raw Signal')

            # Filter and plot selected frequency band
            filtered_data = self.filter_frequency_band(eeg_data, self.current_freq_band)
            color = ['purple', 'brown', 'cyan', 'magenta', 'yellow'][list(self.freq_bands.keys()).index(self.current_freq_band)]
            self.ax1.plot(t, filtered_data[0], color=color, linewidth=2, label=f'{self.current_freq_band} Band')

            # Plot frequency band power in subplots
            if hasattr(self, 'freq_axes'):
                bands = list(self.freq_bands.keys())
                for i, band in enumerate(bands[:len(self.freq_axes)]):
                    filtered = self.filter_frequency_band(eeg_data, band)
                    power = np.abs(filtered[0])  # Simple power estimate
                    self.freq_axes[i].clear()
                    self.freq_axes[i].plot(t, power, color=['purple', 'brown', 'cyan', 'magenta', 'yellow'][i])
                    self.freq_axes[i].set_title(f'{band} Power', fontsize=10)
                    self.freq_axes[i].set_xlabel('Time (s)', fontsize=8)
                    self.freq_axes[i].set_ylabel('Power', fontsize=8)

        elif self.ica_mode:
            # Show ICA components
            self.ax1.set_title('ICA Components - Realistic EEG Data')

            # Apply ICA to the data
            self.ica_components, reconstructed = self.apply_ica(eeg_data)

            # Plot ICA components with vertical offsets
            offsets = np.arange(len(self.ica_components)) * 100
            for i, component in enumerate(self.ica_components):
                if i < len(self.ica_lines):
                    self.ica_lines[i].set_data(t, component + offsets[i])
                    self.ica_lines[i].set_label(f'ICA {i+1}')

            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)

        else:
            # Standard channel view
            if self.ica_var.get():
                # Show individual channels
                self.ax1.set_title('EEG Channels - Realistic Data')
                offsets = np.array([150, 50, -50, -150])
                colors = ['red', 'blue', 'green', 'orange']
                for i in range(min(len(eeg_data), len(colors))):
                    self.ax1.plot(t, eeg_data[i] + offsets[i], color=colors[i],
                                label=f'{self.ch_names[i]} (Realistic)')
            else:
                # Show raw signal
                self.ax1.set_title('Realistic EEG Signal')
                self.ax1.plot(t, eeg_data[0], 'b-', linewidth=2, label='Realistic EEG')

            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)

        # Update brain diagram if it exists
        if hasattr(self, 'ax2'):
            self.init_brain_diagram()

        self.canvas.draw()

    def create_time_series_from_features(self, features):
        """Create a time series signal from the generated features for visualization"""
        # Use the first 40 features (means and stds) to create a representative signal
        n_time_points = self.n_samples
        time_series = np.zeros((self.n_channels, n_time_points))

        params = self.emotion_params[self.current_emotion]
        t = np.linspace(0, self.duration, n_time_points)

        for ch_idx, ch_name in enumerate(self.ch_names):
            # Base signal using emotion-specific parameters
            signal = (
                params['alpha_power'] * np.sin(2 * np.pi * params['alpha_freq'] * t) +
                params['beta_power'] * np.sin(2 * np.pi * params['beta_freq'] * t) +
                params['theta_power'] * np.sin(2 * np.pi * params['theta_freq'] * t) +
                params['delta_power'] * np.sin(2 * np.pi * params['delta_freq'] * t)
            )

            # Modulate signal based on generated features
            if len(features) > ch_idx * 2:
                # Use mean and std features to modulate the signal
                mean_feature = features[ch_idx * 2]
                std_feature = features[ch_idx * 2 + 1]

                # Scale and shift the signal based on features
                signal = signal * (std_feature / 50) + (mean_feature / 10)

            # Add realistic noise
            noise = np.random.normal(0, np.abs(signal).std() * 0.1)
            signal += noise

            time_series[ch_idx] = signal

        return time_series

    def update_plots_with_realistic_data(self, features, time_series):
        """Update plots using the realistic EEG data"""
        t = np.linspace(0, self.duration, self.n_samples)

        # Clear previous plots
        self.ax1.clear()
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Amplitude')

        if self.freq_band_mode:
            # Show frequency band analysis
            self.ax1.set_title(f'EEG Signal - {self.current_freq_band} Band Analysis (Realistic Data)')

            # Plot the time series
            self.ax1.plot(t, time_series[0], 'k-', alpha=0.5, label='Realistic EEG Signal')

            # Filter and plot selected frequency band
            filtered_data = self.filter_frequency_band(time_series, self.current_freq_band)
            color = ['purple', 'brown', 'cyan', 'magenta', 'yellow'][list(self.freq_bands.keys()).index(self.current_freq_band)]
            self.ax1.plot(t, filtered_data[0], color=color, linewidth=2, label=f'{self.current_freq_band} Band')

        elif self.ica_mode:
            # Show ICA components
            self.ax1.set_title('ICA Components - Realistic EEG Data')

            # Apply ICA to the time series
            self.ica_components, reconstructed = self.apply_ica(time_series)

            # Plot ICA components
            offsets = np.arange(len(self.ica_components)) * 100
            for i, component in enumerate(self.ica_components):
                if i < len(self.ica_lines):
                    self.ica_lines[i].set_data(t, component + offsets[i])
                    self.ica_lines[i].set_label(f'ICA {i+1}')

            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)

        else:
            # Standard channel view
            if self.ica_var.get():
                # Show individual channels
                self.ax1.set_title('EEG Channels - Realistic Data')
                offsets = np.array([150, 50, -50, -150])
                colors = ['red', 'blue', 'green', 'orange']
                for i in range(min(len(time_series), len(colors))):
                    self.ax1.plot(t, time_series[i] + offsets[i], color=colors[i],
                                label=f'{self.ch_names[i]} (Realistic)')
            else:
                # Show main signal
                self.ax1.set_title('Realistic EEG Signal')
                self.ax1.plot(t, time_series[0], 'b-', linewidth=2, label='Realistic EEG')

            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)

        # Update brain diagram if it exists
        if hasattr(self, 'ax2'):
            self.init_brain_diagram()

        self.canvas.draw()

    def simulate(self):
        """Main simulation loop using realistic EEG data generation"""
        while self.is_running:
            # Generate realistic EEG data matching training dataset
            eeg_data = self.generate_realistic_eeg_data()

            # For visualization, create a time series from the generated features
            # Use first few features to create a representative signal
            time_series = self.create_time_series_from_features(eeg_data)

            # Save to CSV for model input (simulate the full feature vector)
            feature_names = [f'feature_{i}' for i in range(len(eeg_data))]
            df = pd.DataFrame([eeg_data], columns=feature_names)
            df.to_csv('live_eeg.csv', index=False)

            # Update plots with the time series representation
            self.root.after(0, lambda: self.update_plots_with_realistic_data(eeg_data, time_series))

            self.status_label.config(text=f"Simulating {self.current_emotion} - Realistic data generated")
            time.sleep(1)  # Update every second

    def start_simulation(self):
        if not self.is_running:
            self.is_running = True
            self.sim_thread = threading.Thread(target=self.simulate)
            self.sim_thread.start()
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Running")

    def stop_simulation(self):
        if self.is_running:
            self.is_running = False
            self.sim_thread.join()
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Stopped")

if __name__ == "__main__":
    root = tk.Tk()
    simulator = EEGSimulator(root)
    root.mainloop()

import tkinter as tk
import numpy as np
import threading
import time
import csv
import scipy.stats
from dataclasses import dataclass, field
from typing import List, Tuple
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --------------------------------------------------
# Data structures
# --------------------------------------------------
@dataclass
class EEGSource:
    kind: str                 # 'delta','theta','alpha','beta','gamma','artifact'
    freq: float               # Hz (ignored for certain artifact types)
    amp: float                # amplitude (µV)
    phase: float              # radians
    pos: Tuple[float, float]  # (x,y) in head coords (-1..1)
    decay: float = 0.0        # for transient artifacts
    t_alive: float = 0.0      # internal timer

    def generate(self, t_global: float, t_vec: np.ndarray) -> np.ndarray:
        # Base sinusoidal
        if self.kind in ['delta','theta','alpha','beta','gamma']:
            return self.amp * np.sin(2*np.pi*self.freq*(t_global + t_vec) + self.phase)
        # Artifact types
        if self.kind == 'artifact':
            # Blend of blink transient + drift + line noise + random EMG burst
            sig = 0.0
            # slow drift
            sig += 0.3 * self.amp * np.sin(2*np.pi*0.2*(t_global + t_vec))
            # line noise
            sig += 0.15 * self.amp * np.sin(2*np.pi*50*(t_global + t_vec))
            # occasional blink (exponential) at start of chunk if phase criteria
            if np.random.rand() < 0.03:
                L = min(64, len(t_vec))
                blink = self.amp * 3.0 * np.exp(-np.arange(L)/10)
                pad = np.zeros(len(t_vec))
                pad[:L] += blink
                sig += pad
            # EMG burst
            if np.random.rand() < 0.02:
                emg = np.random.randn(len(t_vec))
                # bandlimit approx by smoothing
                emg = np.convolve(emg, np.ones(5)/5, mode='same')
                sig += 0.8 * self.amp * emg
            return sig
        return np.zeros_like(t_vec)

# --------------------------------------------------
# Simulator
# --------------------------------------------------
class EEGSimulator:
    """EEG simulator emulating interactive concepts of the reference web demo:
    - Presets (Relaxed, Active Task, Noisy, Complex Mix, Manual)
    - Manual source placement (click on head diagram)
    - Modes: RAW / PCA / ICA
    - Noise and Speed sliders
    - Continuous scrolling per-channel traces + component panel
    """
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Interactive Simulator (Raw / PCA / ICA)")
        self.root.geometry("1350x880")

        # Core timing
        self.sampling_rate = 256
        self.update_interval_ms = 80  # ~12.5 FPS base
        self.samples_per_update_base = int(self.sampling_rate * self.update_interval_ms / 1000)
        self.display_duration = 10.0  # seconds history
        self.window_duration = 2.0    # window for PCA/ICA

        # Channels (subset of 10-20)
        self.ch_names = ['Fp1','Fp2','C3','C4']
        self.n_channels = len(self.ch_names)
        self.electrode_positions = {
            'Fp1': (-0.6, 0.8),
            'Fp2': ( 0.6, 0.8),
            'C3' : (-0.5, 0.1),
            'C4' : ( 0.5, 0.1),
        }

        # Buffers
        self.display_samples = int(self.display_duration * self.sampling_rate)
        self.window_samples  = int(self.window_duration  * self.sampling_rate)
        self.display_buffer = np.zeros((self.n_channels, self.display_samples))
        self.window_buffer  = np.zeros((self.n_channels, self.window_samples))
        self.global_time = 0.0
        self.is_running = False

        # Controls state
        self.current_preset = tk.StringVar(value="Relaxed")
        self.view_mode = tk.StringVar(value="RAW")  # RAW | PCA | ICA
        self.noise_level = tk.DoubleVar(value=0.15)  # 0..1 scale
        self.speed_scale = tk.DoubleVar(value=1.0)   # 0.25..2.0

        # Sources
        self.sources: List[EEGSource] = []
        self.manual_mode = False
        self.max_manual_sources = 5

        # Feature accumulation (saved on stop)
        self.accumulated_features = []

        # Build UI & plots
        self._build_controls()
        self._build_figure()

        # Initialize first preset
        self._apply_preset('Relaxed')

    # --------------------------------------------------
    # UI
    # --------------------------------------------------
    def _build_controls(self):
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, pady=4)

        tk.Label(top, text="Preset:").pack(side=tk.LEFT, padx=4)
        presets = ["Relaxed", "Active Task", "Noisy", "Complex Mix", "Manual"]
        tk.OptionMenu(top, self.current_preset, *presets, command=self._on_preset_change).pack(side=tk.LEFT)

        tk.Label(top, text="View:").pack(side=tk.LEFT, padx=8)
        for mode in ["RAW","PCA","ICA"]:
            tk.Radiobutton(top, text=mode, value=mode, variable=self.view_mode, command=self._refresh_mode_label).pack(side=tk.LEFT)

        tk.Label(top, text="Noise:").pack(side=tk.LEFT, padx=8)
        tk.Scale(top, from_=0.0, to=1.0, orient=tk.HORIZONTAL, resolution=0.01, length=120, variable=self.noise_level).pack(side=tk.LEFT)

        tk.Label(top, text="Speed:").pack(side=tk.LEFT, padx=8)
        tk.Scale(top, from_=0.25, to=2.0, orient=tk.HORIZONTAL, resolution=0.05, length=150, variable=self.speed_scale).pack(side=tk.LEFT)

        self.start_btn = tk.Button(top, text="Start", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=10)
        self.stop_btn = tk.Button(top, text="Stop", state=tk.DISABLED, command=self.stop)
        self.stop_btn.pack(side=tk.LEFT)

        self.status_label = tk.Label(self.root, text="Status: Stopped (Preset: Relaxed, Mode: RAW)")
        self.status_label.pack(pady=3)

        self.info_label = tk.Label(self.root, text="Click inside head (Manual preset) to add sources (max 5).")
        self.info_label.pack(pady=2)

    def _refresh_mode_label(self):
        self.status_label.config(text=f"Status: {'Running' if self.is_running else 'Stopped'} (Preset: {self.current_preset.get()}, Mode: {self.view_mode.get()})")

    # --------------------------------------------------
    # Plot layout
    # --------------------------------------------------
    def _build_figure(self):
        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(self.n_channels + 1, 2, width_ratios=[4,1.5], height_ratios=[1]*self.n_channels + [0.9], hspace=0.1, wspace=0.25)

        # Channel axes
        self.channel_axes = []
        self.channel_lines = []
        for i, ch in enumerate(self.ch_names):
            ax = self.fig.add_subplot(gs[i, 0])
            ax.set_xlim(-self.display_duration, 0)
            ax.set_ylabel(ch, rotation=0, labelpad=25)
            if i < self.n_channels - 1:
                ax.set_xticklabels([])
            ax.grid(alpha=0.25)
            (line,) = ax.plot([], [], linewidth=1.0)
            self.channel_axes.append(ax)
            self.channel_lines.append(line)
        self.channel_axes[0].set_title('Real-Time EEG')
        self.channel_axes[-1].set_xlabel('Time (s)')

        # Component axis (PCA/ICA) bottom-left
        self.comp_ax = self.fig.add_subplot(gs[-1, 0])
        self.comp_ax.set_title('Components')
        self.comp_ax.set_xlim(-self.window_duration, 0)
        self.comp_ax.set_yticks([])

        # Head / source placement axis (right column spanning)
        self.head_ax = self.fig.add_subplot(gs[:, 1])
        self._init_head_plot()
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_head_click)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _init_head_plot(self):
        ax = self.head_ax
        ax.clear()
        ax.set_title('Head Diagram / Sources')
        ax.set_xlim(-1,1)
        ax.set_ylim(-0.4,1.05)
        ax.set_aspect('equal')
        ax.axis('off')
        head_circle = plt.Circle((0,0.3),0.95,facecolor='#ddeeff',edgecolor='k',alpha=0.4)
        ax.add_patch(head_circle)
        # Electrodes
        for ch,(x,y) in self.electrode_positions.items():
            ax.add_patch(plt.Circle((x,y),0.07,facecolor='orange',edgecolor='k'))
            ax.text(x,y+0.11,ch,ha='center',va='center',fontsize=9,fontweight='bold')
        # Existing sources
        if hasattr(self,'sources'):
            for s in self.sources:
                ax.add_patch(plt.Circle(s.pos,0.06,facecolor=self._color_for_kind(s.kind),edgecolor='k',alpha=0.8))
                ax.text(s.pos[0], s.pos[1]-0.09, s.kind, ha='center', va='center', fontsize=7)
        ax.text(0,-0.35,'Modes: RAW (mixed), PCA (variance axes), ICA (independent components)',ha='center',fontsize=9)
        self.fig.canvas.draw_idle()

    def _color_for_kind(self, kind: str) -> str:
        mapping = {
            'delta':'#5e3c99','theta':'#1b6996','alpha':'#199e78','beta':'#d98d2f','gamma':'#d73027','artifact':'#555555'
        }
        return mapping.get(kind,'gray')

    # --------------------------------------------------
    # Presets / sources
    # --------------------------------------------------
    def _on_preset_change(self, value):
        self._apply_preset(value)
        self._refresh_mode_label()

    def _apply_preset(self, name: str):
        self.sources.clear()
        self.manual_mode = (name == 'Manual')
        rng = np.random.default_rng()
        if name == 'Relaxed':
            # Alpha dominant frontal & central
            self.sources.append(EEGSource('alpha',10.0,40,rng.uniform(0,2*np.pi),(-0.2,0.6)))
            self.sources.append(EEGSource('alpha',10.0,38,rng.uniform(0,2*np.pi),(0.2,0.6)))
            self.sources.append(EEGSource('theta',6.0,18,rng.uniform(0,2*np.pi),(0.0,0.2)))
        elif name == 'Active Task':
            self.sources.append(EEGSource('beta',20.0,35,rng.uniform(0,2*np.pi),(-0.3,0.5)))
            self.sources.append(EEGSource('beta',20.0,32,rng.uniform(0,2*np.pi),(0.3,0.5)))
            self.sources.append(EEGSource('gamma',40.0,25,rng.uniform(0,2*np.pi),(0.0,0.3)))
            self.sources.append(EEGSource('alpha',11.0,18,rng.uniform(0,2*np.pi),(0.0,0.6)))
        elif name == 'Noisy':
            self.sources.append(EEGSource('alpha',10.0,25,rng.uniform(0,2*np.pi),(0.0,0.55)))
            for _ in range(2):
                self.sources.append(EEGSource('artifact',0.0,40,rng.uniform(0,2*np.pi),(rng.uniform(-0.4,0.4), rng.uniform(0.0,0.6))))
        elif name == 'Complex Mix':
            self.sources.append(EEGSource('delta',2.5,28,rng.uniform(0,2*np.pi),(-0.4,0.2)))
            self.sources.append(EEGSource('theta',6.0,20,rng.uniform(0,2*np.pi),(0.4,0.2)))
            self.sources.append(EEGSource('alpha',10.5,30,rng.uniform(0,2*np.pi),(0.0,0.55)))
            self.sources.append(EEGSource('beta',22.0,22,rng.uniform(0,2*np.pi),(0.2,0.35)))
            self.sources.append(EEGSource('gamma',40.0,16,rng.uniform(0,2*np.pi),(-0.2,0.35)))
        elif name == 'Manual':
            # start empty, user adds
            pass
        self._init_head_plot()

    def _on_head_click(self, event):
        if not self.manual_mode:
            return
        if event.inaxes != self.head_ax:
            return
        if len(self.sources) >= self.max_manual_sources:
            return
        x,y = event.xdata, event.ydata
        # cycle waveform types
        order = ['delta','theta','alpha','beta','gamma','artifact']
        next_kind = order[len(self.sources) % len(order)]
        freq_map = {'delta':2.5,'theta':6,'alpha':10,'beta':20,'gamma':40,'artifact':0}
        amp_map  = {'delta':25,'theta':20,'alpha':35,'beta':28,'gamma':20,'artifact':35}
        phase = np.random.uniform(0,2*np.pi)
        self.sources.append(EEGSource(next_kind,freq_map[next_kind],amp_map[next_kind],phase,(x,y)))
        self._init_head_plot()

    # --------------------------------------------------
    # Signal synthesis
    # --------------------------------------------------
    def _distance_weight_matrix(self):
        # weight of each source to each channel via Gaussian falloff
        sigma = 0.55
        w = np.zeros((self.n_channels, len(self.sources)))
        for ci,ch in enumerate(self.ch_names):
            cx,cy = self.electrode_positions[ch]
            for si,src in enumerate(self.sources):
                dx = cx - src.pos[0]
                dy = cy - src.pos[1]
                d2 = dx*dx + dy*dy
                w[ci,si] = np.exp(-d2/(2*sigma*sigma))
        # normalize per source to keep energy stable
        if w.size:
            w /= (np.max(w, axis=0, keepdims=True) + 1e-9)
        return w

    def _generate_chunk(self, n_samples: int):
        if not self.sources:
            return np.zeros((self.n_channels, n_samples))
        t_vec = np.arange(n_samples) / self.sampling_rate
        w = self._distance_weight_matrix()  # (channels x sources)
        src_signals = []
        for s in self.sources:
            sig = s.generate(self.global_time, t_vec)
            src_signals.append(sig)
        S = np.array(src_signals)  # sources x samples
        mixed = w @ S
        # Add gaussian + pink-ish noise
        noise_amp = self.noise_level.get()*10
        if noise_amp > 0:
            white = np.random.randn(*mixed.shape)
            pink = np.cumsum(white, axis=1)
            pink /= (np.std(pink, axis=1, keepdims=True)+1e-9)
            mixed += noise_amp*0.6*white + noise_amp*0.4*pink
        return mixed

    # --------------------------------------------------
    # Feature extraction (simple demo features for saving)
    # --------------------------------------------------
    def _extract_features(self):
        data = self.window_buffer
        feats = []
        for ch in range(self.n_channels):
            x = data[ch]
            feats.extend([
                np.mean(x), np.std(x), np.var(x), np.min(x), np.max(x), np.ptp(x),
                scipy.stats.skew(x), scipy.stats.kurtosis(x)
            ])
            freqs = np.fft.rfftfreq(len(x), 1/self.sampling_rate)
            spec = np.abs(np.fft.rfft(x))**2
            bands = [(1,4),(4,8),(8,13),(13,30),(30,45)]
            for lo,hi in bands:
                mask = (freqs>=lo)&(freqs<hi)
                feats.append(float(np.sum(spec[mask])))
        return feats

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    def _update_plots(self):
        # Channel traces
        t_disp = np.linspace(-self.display_duration,0,self.display_samples)
        # View mode adjustments
        view = self.view_mode.get()
        window_data = self.window_buffer.copy()
        if view == 'PCA' and self.window_buffer.shape[1] > 10:
            # reorder channels by power
            powers = np.sum(self.window_buffer**2, axis=1)
            order = np.argsort(-powers)
            window_data = window_data[order]
            disp_data = self.display_buffer[order]
            name_order = [self.ch_names[i] for i in order]
        else:
            disp_data = self.display_buffer
            name_order = self.ch_names
        for i, line in enumerate(self.channel_lines):
            ch_data = disp_data[i]
            line.set_data(t_disp, ch_data)
            ax = self.channel_axes[i]
            ax.set_ylabel(name_order[i], rotation=0, labelpad=25)
            q = np.percentile(ch_data,[5,95])
            if q[0]==q[1]:
                q=[q[0]-1,q[1]+1]
            margin = 0.15*(q[1]-q[0])
            ax.set_ylim(q[0]-margin, q[1]+margin)
        # Component panel
        self.comp_ax.clear()
        self.comp_ax.set_xlim(-self.window_duration,0)
        self.comp_ax.set_yticks([])
        t_win = np.linspace(-self.window_duration,0,self.window_samples)
        if view == 'ICA' and self.window_buffer.shape[1] > 30:
            try:
                ica = FastICA(n_components=min(self.n_channels,4), random_state=0, max_iter=400)
                S = ica.fit_transform(window_data.T).T
                offs = np.arange(S.shape[0])*60
                for k in range(S.shape[0]):
                    self.comp_ax.plot(t_win, S[k]+offs[k], label=f'IC{k+1}')
                self.comp_ax.set_title('ICA Components')
            except Exception:
                self.comp_ax.set_title('ICA Failed')
        elif view == 'PCA' and self.window_buffer.shape[1] > 30:
            try:
                pca = PCA(n_components=min(self.n_channels,4))
                S = pca.fit_transform(window_data.T).T
                offs = np.arange(S.shape[0])*60
                for k in range(S.shape[0]):
                    self.comp_ax.plot(t_win, S[k]+offs[k], label=f'PC{k+1}')
                self.comp_ax.set_title('PCA Components')
            except Exception:
                self.comp_ax.set_title('PCA Failed')
        else:
            # Raw summary (RMS per channel)
            rms = np.sqrt(np.mean(window_data**2, axis=1))
            self.comp_ax.bar(range(len(rms)), rms, color='#8888cc')
            self.comp_ax.set_xticks(range(len(rms)))
            self.comp_ax.set_xticklabels(name_order)
            self.comp_ax.set_title('Channel RMS (RAW)')
        self.comp_ax.grid(alpha=0.25)
        self._init_head_plot()  # refresh sources overlay
        self.canvas.draw_idle()

    # --------------------------------------------------
    # Main streaming loop
    # --------------------------------------------------
    def _loop(self):
        while self.is_running:
            start = time.time()
            speed = self.speed_scale.get()
            n = max(1, int(self.samples_per_update_base * speed))
            chunk = self._generate_chunk(n)
            # Update buffers (ring via roll)
            if n >= self.display_samples:
                self.display_buffer = chunk[:, -self.display_samples:]
            else:
                self.display_buffer = np.roll(self.display_buffer, -n, axis=1)
                self.display_buffer[:, -n:] = chunk
            if n >= self.window_samples:
                self.window_buffer = chunk[:, -self.window_samples:]
            else:
                self.window_buffer = np.roll(self.window_buffer, -n, axis=1)
                self.window_buffer[:, -n:] = chunk
            self.global_time += n / self.sampling_rate
            # Features (store periodically e.g. every 0.5 s)
            if int(self.global_time*2) != int((self.global_time - n/self.sampling_rate)*2):
                self.accumulated_features.append(self._extract_features())
            # Update plots less often if speed high
            if int(self.global_time*5) != int((self.global_time - n/self.sampling_rate)*5):
                self.root.after(0, self._update_plots)
            # Sleep to maintain approximate realtime feel
            elapsed = (time.time()-start)
            target = self.update_interval_ms/1000.0 / max(speed,0.1)
            to_sleep = max(0.0, target - elapsed)
            time.sleep(to_sleep)
        # finalize

    # --------------------------------------------------
    # Control
    # --------------------------------------------------
    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"Status: Running (Preset: {self.current_preset.get()}, Mode: {self.view_mode.get()})")
        self.accumulated_features.clear()
        self.global_time = 0.0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text=f"Status: Stopped (Preset: {self.current_preset.get()}, Mode: {self.view_mode.get()})")
        # Save accumulated features
        if self.accumulated_features:
            csv_file = 'realtime_eeg_data.csv'
            with open(csv_file,'w',newline='') as f:
                writer = csv.writer(f)
                # simple header (not full 2548, just dynamic length)
                n_feat = len(self.accumulated_features[0])
                writer.writerow([f'f{i}' for i in range(n_feat)])
                writer.writerows(self.accumulated_features)
            print(f"Saved {len(self.accumulated_features)} feature rows to {csv_file}")
        else:
            print("No features captured.")

if __name__ == '__main__':
    root = tk.Tk()
    app = EEGSimulator(root)
    root.mainloop()

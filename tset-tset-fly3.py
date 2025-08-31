import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import convolve2d
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

# --- 1. The Neuron (Unchanged) ---
class LearningNeuron:
    """A simple neuron that spikes and has a recovery variable."""
    def __init__(self):
        self.voltage = np.random.uniform(-70, -50)
        self.recovery = 0.2 * self.voltage
        self.fired = False

    def update(self, dt, I_stim, I_ephaptic, I_synaptic, I_theta):
        self.fired = False
        if self.voltage >= 30:
            self.voltage = -65
            self.recovery += 8
            self.fired = True

        total_current = I_stim + I_ephaptic + I_synaptic + I_theta
        dv = 0.04 * self.voltage**2 + 5 * self.voltage + 140 - self.recovery + total_current
        self.voltage += dv * dt
        self.recovery += 0.02 * (0.2 * self.voltage - self.recovery) * dt
        self.voltage = np.clip(self.voltage, -100, 30)

# --- 2. The Neural Field with Holographic Enhancements ---
class HolographicField:
    """A neural field with mechanisms to promote holographic memory."""
    def __init__(self, size=(50, 50)):
        self.size = size
        self.neurons = np.array([[LearningNeuron() for _ in range(size[1])] for _ in range(size[0])])
        
        self.synaptic_weights = np.random.uniform(0.01, 0.1, (size[0], size[1], 3, 3))
        self.synaptic_weights[:, :, 1, 1] = 0

        # Simulation & Learning Parameters
        self.dt = 0.5
        self.ephaptic_strength = 2.5
        self.learning_rate = 0.002
        
        # --- Holographic Upgrades ---
        self.theta_phase_jitter = 0.5  # Creates phase-multiplexed input
        self.weight_decay = 1e-5       # Prevents weights from saturating
        self.weight_diffusion = 1e-4   # Spreads learning traces globally

        self.theta_phase = 0.0

    def step(self, stimulus_grid):
        voltages = np.array([[n.voltage for n in row] for row in self.neurons])
        firings_prev = np.array([[n.fired for n in row] for row in self.neurons])

        # 1. Ephaptic Coupling (Physics)
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) / 8.0
        avg_neighbor_voltage = convolve2d(voltages, kernel, mode='same', boundary='wrap')
        I_ephaptic = self.ephaptic_strength * (avg_neighbor_voltage - voltages)

        # 2. Synaptic Input (Learned)
        I_synaptic = np.zeros(self.size)
        # (Loop for synaptic input remains the same)
        for r_off in range(-1, 2):
            for c_off in range(-1, 2):
                if r_off == 0 and c_off == 0: continue
                neighbor_firings = np.roll(firings_prev, (-r_off, -c_off), axis=(0, 1))
                I_synaptic += self.synaptic_weights[:, :, r_off+1, c_off+1] * neighbor_firings * 20.0

        # 3. Global Theta (with Jitter for Holography)
        theta_freq_hz = 8.0
        self.theta_phase += 2 * np.pi * (theta_freq_hz / 1000.0) * self.dt
        jittered_phase = self.theta_phase + np.random.normal(0, self.theta_phase_jitter, self.size)
        I_theta = 7.0 * np.sin(jittered_phase)

        # Update neurons
        for r in range(self.size[0]):
            for c in range(self.size[1]):
                self.neurons[r, c].update(self.dt, stimulus_grid[r, c], I_ephaptic[r, c], I_synaptic[r, c], I_theta[r, c])
        
        # 4. Hebbian Learning (with Decay and Diffusion for Holography)
        firings_now = np.array([[n.fired for n in row] for row in self.neurons])
        # (Hebbian update loop remains the same)
        for r_off in range(-1, 2):
            for c_off in range(-1, 2):
                if r_off == 0 and c_off == 0: continue
                neighbor_firings = np.roll(firings_now, (-r_off, -c_off), axis=(0, 1))
                self.synaptic_weights[:, :, r_off+1, c_off+1] += self.learning_rate * firings_now * neighbor_firings
        
        # Apply decay
        self.synaptic_weights *= (1 - self.weight_decay)

        # Apply diffusion
        mean_weights = self.get_synaptic_strength_grid()
        diffusion_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) # Laplacian kernel
        diffusion_effect = convolve2d(mean_weights, diffusion_kernel, mode='same', boundary='wrap')
        self.synaptic_weights += self.weight_diffusion * diffusion_effect[:, :, np.newaxis, np.newaxis]

        self.synaptic_weights = np.clip(self.synaptic_weights, 0, 1.0)

    def get_voltage_grid(self):
        return np.array([[n.voltage for n in row] for row in self.neurons])
    
    def get_synaptic_strength_grid(self):
        return np.mean(self.synaptic_weights, axis=(2, 3))

# --- 3. The Application with Live Holographic Analysis ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Holographic Learning Field")
        
        self.field_size = (50, 50)
        self.display_size = (300, 300)
        
        self.field = HolographicField(self.field_size)
        self.cap = cv2.VideoCapture(0)

        # --- GUI Setup ---
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=10)
        
        self.webcam_label = tk.Label(left_frame, text="Webcam Input")
        self.webcam_label.pack()
        self.webcam_canvas = tk.Canvas(left_frame, width=self.display_size[0], height=self.display_size[1])
        self.webcam_canvas.pack()

        self.activity_label = tk.Label(left_frame, text="Neural Activity (Voltage)")
        self.activity_label.pack()
        self.activity_canvas = tk.Canvas(left_frame, width=self.display_size[0], height=self.display_size[1])
        self.activity_canvas.pack()

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.fig = Figure(figsize=(8, 8))
        self.ax_weights = self.fig.add_subplot(211)
        self.ax_holo = self.fig.add_subplot(212)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.im_weights = self.ax_weights.imshow(self.field.get_synaptic_strength_grid(), cmap='magma', vmin=0, vmax=0.5, animated=True)
        self.ax_weights.set_title("Learned Synaptic Strength (Engram)")
        self.ax_weights.axis('off')

        self.patch_sizes = [5, 10, 15, 20] # in pixels
        self.holo_plot, = self.ax_holo.plot(self.patch_sizes, [0]*len(self.patch_sizes), 'o-', label='Reconstruction Quality')
        self.ax_holo.set_title("Live Holographic Memory Test")
        self.ax_holo.set_xlabel("Patch Size Used for Reconstruction")
        self.ax_holo.set_ylabel("Correlation with Full Engram")
        self.ax_holo.set_ylim(-0.1, 1.0)
        self.ax_holo.grid(True, linestyle='--')

        self.frame_count = 0
        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            # (Webcam processing and neural update are the same as before)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_for_display = Image.fromarray(frame_rgb).resize(self.display_size)
            self.photo_webcam = ImageTk.PhotoImage(image=img_for_display)
            self.webcam_canvas.create_image(0, 0, image=self.photo_webcam, anchor=tk.NW)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            stimulus_grid = cv2.resize(gray_frame, self.field_size, interpolation=cv2.INTER_AREA)
            stimulus_grid = (stimulus_grid / 255.0) * 15.0
            self.field.step(stimulus_grid)
            
            # Visualize Activity and Engram
            voltages = self.field.get_voltage_grid()
            activity_img = self.mat_to_image(voltages, cmap='viridis', vmin=-70, vmax=30)
            self.photo_activity = ImageTk.PhotoImage(image=activity_img)
            self.activity_canvas.create_image(0, 0, image=self.photo_activity, anchor=tk.NW)
            
            weights = self.field.get_synaptic_strength_grid()
            self.im_weights.set_data(weights)
            
            # --- Run Holographic Analysis Periodically ---
            if self.frame_count % 50 == 0: # Run test every 50 frames
                self.update_holographic_analysis(weights)

            self.canvas.draw()
            self.frame_count += 1
        
        self.root.after(30, self.update)

    def update_holographic_analysis(self, engram):
        """Reconstruct the engram from patches of different sizes."""
        correlations = []
        R, C = engram.shape
        grid_y, grid_x = np.mgrid[0:R, 0:C]

        for p_size in self.patch_sizes:
            if R > p_size and C > p_size:
                start_r, start_c = (R - p_size) // 2, (C - p_size) // 2
                patch = engram[start_r:start_r+p_size, start_c:start_c+p_size]

                ys, xs = np.mgrid[start_r:start_r+p_size, start_c:start_c+p_size]
                
                try:
                    rbf = Rbf(xs.ravel(), ys.ravel(), patch.ravel(), function='multiquadric', smooth=0.1)
                    recon = rbf(grid_x, grid_y)
                    
                    # Calculate correlation
                    corr = np.corrcoef(engram.ravel(), recon.ravel())[0, 1]
                    correlations.append(corr if not np.isnan(corr) else 0)
                except Exception:
                    correlations.append(0) # Rbf can fail if data is flat
            else:
                correlations.append(0)
        
        self.holo_plot.set_ydata(correlations)
        
    def mat_to_image(self, mat, cmap='viridis', vmin=None, vmax=None):
        norm_mat = plt.Normalize(vmin=vmin, vmax=vmax)(mat)
        colored_mat = plt.get_cmap(cmap)(norm_mat)[:, :, :3]
        img_uint8 = (colored_mat * 255).astype(np.uint8)
        return Image.fromarray(img_uint8).resize(self.display_size)

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
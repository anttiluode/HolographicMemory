"""
VAE Transplant Experiment

This script tests the hypothesis that a VAE's learned knowledge is distributed
in a holographic-like manner.

It performs a three-phase experiment:
1.  Train a "Donor" VAE on a live webcam feed to create a memory engram.
2.  "Transplant" a small piece of the Donor's brain (the first convolutional layer)
    into a new, untrained "Recipient" VAE.
3.  "Race" the Recipient VAE against a "Control" VAE (trained from scratch)
    to see if the transplant provides a significant learning advantage.

The expected outcome is that the Recipient will learn faster (its loss will
decrease more quickly), demonstrating that even a small piece of a trained
network contains a meaningful trace of the whole.
"""

import os
import sys
import types
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
import time
import matplotlib.pyplot as plt

# ===============================
# Environment Setup & Monkey-Patch for Triton
# This patch MUST run before importing diffusers.
# ===============================
os.environ["DIFFUSERS_NO_IP_ADAPTER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    import triton.runtime
except ImportError:
    sys.modules["triton"] = types.ModuleType("triton")
    sys.modules["triton.runtime"] = types.ModuleType("triton.runtime")
    import triton.runtime

if not hasattr(triton.runtime, "Autotuner"):
    class DummyAutotuner:
        def __init__(self, *args, **kwargs):
            pass
        def tune(self, *args, **kwargs):
            return None
    triton.runtime.Autotuner = DummyAutotuner

# ===============================
# NOW it is safe to import diffusers
# ===============================
from diffusers import StableVideoDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# VAE Components (from unitvae4.py)
# ===============================
class AdaptiveEncoderConv(nn.Module):
    def __init__(self):
        super(AdaptiveEncoderConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return self.conv4(x)

class AdaptiveDecoderConv(nn.Module):
    def __init__(self):
        super(AdaptiveDecoderConv, self).__init__()
        self.conv_trans1 = nn.ConvTranspose2d(4, 256, kernel_size=3, stride=1, padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv_trans3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv_trans4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
    def forward(self, latent):
        x = self.relu(self.conv_trans1(latent))
        x = self.relu(self.conv_trans2(x))
        x = self.relu(self.conv_trans3(x))
        return torch.sigmoid(self.conv_trans4(x))

class AdaptiveVAETrainer:
    def __init__(self, encoder, decoder, teacher_vae):
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_vae = teacher_vae
        self.optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
        self.loss_fn = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler()
        
    def train_on_frame(self, image_tensor):
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()
        with torch.no_grad():
            teacher_latent = self.teacher_vae.encode(image_tensor.half()).latent_dist.sample().float()
            decoded = self.teacher_vae.decode(teacher_latent.half(), num_frames=1).sample
            teacher_decoded = ((decoded / 2 + 0.5).clamp(0, 1)).float()
        with torch.cuda.amp.autocast():
            pred_latent = self.encoder(image_tensor)
            pred_image = self.decoder(pred_latent)
            loss = self.loss_fn(pred_latent, teacher_latent) + self.loss_fn(pred_image, teacher_decoded)
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

# ===============================
# Main Experiment Logic
# ===============================
def run_experiment():
    # 1. SETUP
    print("Loading Teacher VAE...")
    video_pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16
    ).to(device)
    transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # 2. PHASE 1: Train the Donor Brain
    print("\n--- PHASE 1: Training Donor Brain (100 steps) ---")
    donor_encoder = AdaptiveEncoderConv().to(device)
    donor_decoder = AdaptiveDecoderConv().to(device)
    donor_trainer = AdaptiveVAETrainer(donor_encoder, donor_decoder, video_pipe.vae)
    
    for i in range(100):
        ret, frame = cap.read()
        if not ret: continue
        image_tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        loss = donor_trainer.train_on_frame(image_tensor)
        if (i+1) % 20 == 0:
            print(f"Donor training step {i+1}/100, Loss: {loss:.4f}")
    
    donor_weights = donor_encoder.state_dict()
    print("✅ Donor Brain trained.")

    # 3. PHASE 2: The Transplant
    print("\n--- PHASE 2: Performing Transplant ---")
    
    recipient_encoder = AdaptiveEncoderConv().to(device)
    recipient_decoder = AdaptiveDecoderConv().to(device)
    control_encoder = AdaptiveEncoderConv().to(device)
    control_decoder = AdaptiveDecoderConv().to(device)

    # Perform the transplant of the first conv layer's weights and biases
    transplant_weights = {k: v for k, v in donor_weights.items() if 'conv1' in k}
    recipient_encoder_dict = recipient_encoder.state_dict()
    recipient_encoder_dict.update(transplant_weights)
    
    # Use strict=False because we are intentionally loading only a partial state_dict
    recipient_encoder.load_state_dict(recipient_encoder_dict, strict=False)
    
    print(f"✅ Transplanted {len(transplant_weights.keys())} weight tensors into Recipient Brain's first layer.")

    # 4. PHASE 3: The Race
    print("\n--- PHASE 3: Training Recipient vs. Control (100 steps) ---")
    recipient_trainer = AdaptiveVAETrainer(recipient_encoder, recipient_decoder, video_pipe.vae)
    control_trainer = AdaptiveVAETrainer(control_encoder, control_decoder, video_pipe.vae)
    
    recipient_losses, control_losses = [], []
    steps = []

    for i in range(100):
        ret, frame = cap.read()
        if not ret: continue
        image_tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)

        loss_r = recipient_trainer.train_on_frame(image_tensor)
        loss_c = control_trainer.train_on_frame(image_tensor)

        recipient_losses.append(loss_r)
        control_losses.append(loss_c)
        steps.append(i + 1)

        if (i+1) % 20 == 0:
            print(f"Race step {i+1}/100 -> Recipient Loss: {loss_r:.4f} | Control Loss: {loss_c:.4f}")

    cap.release()
    print("✅ Race finished.")

    # 5. PLOT RESULTS
    plt.figure(figsize=(12, 7))
    plt.plot(steps, control_losses, 'r--', label='Control Brain (From Scratch)')
    plt.plot(steps, recipient_losses, 'g-', label='Recipient Brain (With Transplant)')
    plt.xlabel("Training Steps")
    plt.ylabel("Training Loss (Lower is Better)")
    plt.title("Impact of Transplanting a Piece of a Learned VAE")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()

if __name__ == "__main__":
    run_experiment()
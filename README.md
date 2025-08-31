## Temporal Processing in Neural Systems
This repository contains Python scripts exploring how different neural systems, both artificial and biologically inspired,
process time-ordered information and form distributed memories.

### 1. VAE Brain Transplant (brain_transplant.py)
This script tests the "holographic memory" hypothesis in a modern deep learning model (a Variational Autoencoder).

What it does: It trains a "donor" AI on a live webcam feed, then "transplants" a small piece of its learned knowledge 
(the first layer of weights) into a new "recipient" AI. It then races the recipient against a control AI that learns from scratch.

Key Finding: The transplant gives the recipient a significant head start, demonstrating that memory in these networks 
is distributed, and even a small piece contains a meaningful trace of the whole.

### 2. Holographic Learning Field (tset-tset-fly3.py)
This script simulates a 2D field of biologically inspired neurons that learn from a webcam feed. It models several
key brain mechanisms.

What it does: It visualizes how a network of simple spiking neurons uses ephaptic coupling (local field effects) and 
a global theta oscillation to self-organize. It then uses Hebbian learning to form a persistent memory, or engram, of what it sees.

Key Finding: It includes a live test for "holographic-ness," showing how a distributed memory trace forms over time, 
allowing the full memory to be partially reconstructed from just a small piece.

## Installation
To run these experiments, you will need Python 3.8+ and the following libraries. You can install them all with a single 
pip command.

### Requirements
A CUDA-enabled NVIDIA GPU is strongly recommended, especially for brain_transplant.py.

A webcam is required for both scripts.

### Pip Install Command
Copy and paste the following command into your terminal:

pip install torch torchvision opencv-python numpy pillow matplotlib scipy diffusers transformers ac

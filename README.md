## Project overview

This repository contains the implementation of my thesis project titled **"Creating Non-Equilibrium States with Quantum Generative Models"**.

The main objective of this work is to explore how quantum generative models can be used to construct and sample from **non-equilibrium quantum states**, a regime that is significantly less understood than equilibrium quantum systems and plays an important role in quantum dynamics and information theory.

The core contribution so far is the development of a **Quantum Denoising Diffusion Model (QDDM)** capable of generating **mixed quantum state distributions**. To the best of my knowledge, this represents a novel extension of diffusion-based generative modelling into the mixed-state quantum setting.

Initially, the model has been developed and tested in the simplest non-trivial setting of a **single qubit system**, focusing on validating the framework and its ability to reproduce target mixed-state distributions.

---

## ⚠️ Current structure

At the moment, the repository is organized into three main components:

- `utiles.py` → utility functions used across the project  
- `QDDM.py` → core implementation of the diffusion and denoising model classes  
- `main.ipynb` → example execution of the model, currently running a 1 qubit mixed-state distribution on a ring geometry  

This repository is under active development and does not yet represent the final version of the project. The code is actively being cleaned and reorganized.

Planned extensions include:
- Improving code structure and readability
- Expanding the set of experiments shown in `main.py`, showing more demonstrations of the model's behavior
- Adding the version of the model working with **pure quantum state distributions**
- Showing execution of parts of the model on **simulated and real quantum hardware**
- Extending the framework to **multi-qubit systems**, with the goal of generating non-equilibrium states in higher-dimensional settings  

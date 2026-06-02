import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from qutip import Bloch
import os
from qiskit.visualization import plot_state_qsphere

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def density_matrix_to_ensemble(matrix):
    """
    Diagonalizes a batch of density matrices to obtain their spectral ensemble.
    
    Args:
        matrix (torch.Tensor): Batch of density matrices of shape (N, 2**n, 2**n).
        
    Returns:
        probs (torch.Tensor): Eigenvalues (probabilities) of shape (N, 2**n).
        ensemble (torch.Tensor): Corresponding eigenvectors of shape (N, 2**n, 2**n).
    """
    probs, ensemble = torch.linalg.eigh(matrix)
    return probs, ensemble

def ensemble_to_density_matrix(probs, ensemble):
    """
    Reconstructs density matrices from their eigenvalues and eigenvectors.
    
    Args:
        probs (torch.Tensor): Probabilities (eigenvalues) of shape (N, 2**n).
        ensemble (torch.Tensor): Eigenvectors of shape (N, 2**n, 2**n).
        
    Returns:
        rho (torch.Tensor): Reconstructed density matrices of shape (N, 2**n, 2**n).
    """
    probs_complex = probs.to(ensemble.dtype) # Cast to complex for multiplication
    rho = torch.einsum('nk, nik, njk -> nij', probs_complex, ensemble, ensemble.conj())
    return rho

def time_ensemble_to_time_matrices(N, n, T, time_probs, time_ensemble):
    """
    Converts the entire time history of diffusion ensembles into density matrices.
    
    Args:
        N (int): Number of states per batch.
        n (int): Number of main system qubits.
        T (int): Total number of diffusion timesteps.
        time_probs (torch.Tensor): History of probabilities of shape (N, 2**n, T+1).
        time_ensemble (torch.Tensor): History of eigenvectors of shape (N, 2**n, 2**n, T+1).
        
    Returns:
        time_matrices (torch.Tensor): History of density matrices of shape (N, 2**n, 2**n, T+1).
    """
    time_matrices = torch.zeros([N, 2**n, 2**n, T+1], dtype=torch.complex128)
    for t in range(T+1):
        time_matrices[:,:,:,t] = ensemble_to_density_matrix(time_probs[:,:,t], time_ensemble[:,:,:,t])
    return time_matrices

def _bloch_xyz_mix(rho):
    """
    Calculates the Cartesian coordinates (x, y, z) of the Bloch vector from 1-qubit density matrices.
    
    Args:
        rho (torch.Tensor): Batch of density matrices of shape (N, 2, 2).
        
    Returns:
        x, y, z (torch.Tensor): Individual Cartesian coordinates extracted from the matrices.
    """
    x = rho[:,0,1] + rho[:,1,0]
    y = 1j*(rho[:,0,1] - rho[:,1,0])
    z = rho[:,0,0] - rho[:,1,1]
    return x, y, z

def save_simulation(folder_name, filename, N, L, steps, best_params, best_ensemble, 
                    best_ensemble_probs, loss_hist, best_loss_list, ancilla_rand_state, norm):
    """
    Saves all simulation tensors and metadata into a single PyTorch dictionary file (.pt).
    
    Args:
        folder_name (str): Destination directory path.
        filename (str): Name of the file (e.g., 'simulation_L2_N500_ANILLO_dep.pt').
        N, L, steps (int): Simulation hyperparameters.
        best_params, best_ensemble, best_ensemble_probs, ancilla_rand_state (torch.Tensor): Quantum state tensors.
        loss_hist, best_loss_list (list): Training metrics.
        norm (float): Normalization factor.
    """
    # Create the directory if it doesn't exist to prevent errors
    os.makedirs(folder_name, exist_ok=True)
    
    file_path = os.path.join(folder_name, filename)
    
    # Pack everything securely into a dictionary
    data_dict = {
        'N': N,
        'L': L,
        'steps': steps,
        'best_params': best_params,
        'best_ensemble': best_ensemble,
        'best_ensemble_probs': best_ensemble_probs,
        'loss_hist': loss_hist,
        'best_loss_list': best_loss_list,
        'ancilla_rand_state': ancilla_rand_state,
        'norm': norm
    }
    
    # Save the dictionary native to PyTorch
    torch.save(data_dict, file_path)
    print(f"[SAVE] Simulation successfully saved at: {file_path}")

def load_simulation(folder_name, filename):
    """
    Loads a previously saved simulation dictionary.
    
    Args:
        folder_name (str): Source directory path.
        filename (str): Name of the file to load.
        
    Returns:
        dict: Dictionary containing all the saved simulation tensors and metadata.
    """
    file_path = os.path.join(folder_name, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] The file {file_path} does not exist.")
        
    data_dict = torch.load(file_path)
    print(f"[LOAD] Simulation successfully loaded from: {file_path}")
    
    return data_dict

# =============================================================================
# DISTRIBUTION GENERATION
# =============================================================================

def generate_Haar_distribution(N, n):
    """
    Generates a batch of random pure states following the Haar measure.
    
    Args:
        N (int): Number of states to generate.
        n (int): Number of qubits.
        
    Returns:
        haar_states (torch.Tensor): Normalized state vectors of shape (N, 2**n).
    """
    states = torch.randn(N, 2**n, dtype=torch.float64) + 1j*torch.randn(N, 2**n, dtype=torch.float64)
    norms = torch.linalg.norm(states, dim=1, keepdim=True)
    haar_states = states / norms
    return haar_states

def generate_maximally_mixed_ensemble(N, n):
    """
    Generates a batch of N maximally mixed states (quantum white noise) for n qubits.
    
    Args:
        N (int): Number of states to generate.
        n (int): Number of qubits.
        
    Returns:
        probs (torch.Tensor): Uniform probability distribution.
        ensemble (torch.Tensor): Identity matrix acting as eigenvectors.
    """
    probs = torch.ones([N, 2**n], dtype=torch.float64) / 2**n
    I = torch.eye(2**n, dtype=torch.complex128)
    ensemble = I.unsqueeze(0).repeat(N, 1, 1)
    return probs, ensemble

def circleX(N): 
    """Generates N pure states forming a great circle in the YZ plane (perpendicular to X)."""
    states = torch.zeros([N, 2], dtype=torch.complex128) 
    phis = torch.rand(N) * 2 * torch.pi
    states[:, 0] = torch.cos(phis)
    states[:, 1] = -1j * torch.sin(phis)
    return states

def circleY(N): 
    """Generates N pure states forming a great circle in the XZ plane (perpendicular to Y)."""
    states = torch.zeros([N, 2], dtype=torch.complex128) 
    phis = torch.rand(N) * 2 * torch.pi
    states[:, 0] = torch.cos(phis)
    states[:, 1] = torch.sin(phis)
    return states

def circleZ(N): 
    """Generates N pure states forming a great circle on the equator (XY plane, perpendicular to Z)."""
    states = torch.zeros([N, 2], dtype=torch.complex128) 
    phis = torch.rand(N) * 2 * torch.pi
    r2 = math.sqrt(2.0)
    states[:, 0] = 1/r2
    states[:, 1] = torch.exp(-1j*phis)/r2
    return states

def generate_mix_ring(N, p, A):
    """
    Generates a ring of mixed states by applying a depolarizing channel over a circle of pure states.
    
    Args:
        N (int): Number of states to generate.
        p (float): Noise parameter (0 = pure states, 1 = maximally mixed center).
        A (int): Axis of rotation (0=Z, 1=Y, 2=X).
        
    Returns:
        probs, ensemble (torch.Tensor): Ensemble of the resulting noisy states.
    """
    if A == 0:
        circle_pure = circleZ(N)
    elif A == 1:
        circle_pure = circleY(N)
    elif A == 2:
        circle_pure = circleX(N)

    rho_pure = torch.einsum('ni,nj->nij', circle_pure, circle_pure.conj())
    I = torch.eye(2, dtype=torch.complex128).unsqueeze(0).repeat(N, 1, 1)
    
    circle_mix = (1 - 4*p/3) * rho_pure + 4*p/3 * (I/2) 

    probs, ensemble = density_matrix_to_ensemble(circle_mix)
    return probs, ensemble

def mix_diameterz_ensemble(N):
    """
    Generates N mixed states randomly distributed along the Z-axis diameter of the Bloch sphere.
    
    Args:
        N (int): Number of states.
        
    Returns:
        probs, diameterz_ensemble (torch.Tensor): Resulting ensemble aligned with the Z-axis.
    """
    p0 = torch.rand(N)
    p1 = 1 - p0
    probs = torch.zeros([N, 2], dtype=torch.float64)
    diameterz_ensemble = torch.zeros((N, 2, 2), dtype=torch.complex128) 

    probs[:,0] = p0
    probs[:,1] = p1
    diameterz_ensemble[:,0,0] = 1
    diameterz_ensemble[:,1,1] = 1
    return probs, diameterz_ensemble

def mix_diametery_ensemble(N):
    """
    Generates N mixed states randomly distributed along the Y-axis diameter of the Bloch sphere.
    
    Args:
        N (int): Number of states.
        
    Returns:
        probs, diametery (torch.Tensor): Resulting ensemble aligned with the Y-axis.
    """
    p0 = torch.rand(N)
    probs = torch.zeros([N, 2], dtype=torch.float64)
    diametery = torch.zeros((N, 2, 2), dtype=torch.complex128)
    
    probs[:,0] = p0
    probs[:,1] = 1 - p0

    r = np.sqrt(2)
    diametery[:,0,0] = 1/r
    diametery[:,1,0] = 1j/r
    diametery[:,0,1] = 1/r
    diametery[:,1,1] = -1j/r
    
    return probs, diametery


def amplitude_damping_haar(N, p):
    """
    Generates an amplitude damping ensemble mathematically for 1 qubit.
    
    Args:
        N (int): Number of states.
        p (float): Decay probability (0 = pure, 1 = decayed to |0>).
        
    Returns:
        probs (torch.Tensor): Eigenvalues of shape (N, 2).
        ensemble (torch.Tensor): Eigenvectors of shape (N, 2, 2).
    """
    initial_states=generate_Haar_distribution(N, 1)

    rho_in = torch.einsum('ni,nj->nij', initial_states, initial_states.conj())
    
    rho_out = torch.zeros_like(rho_in)
    
    rho_out[:, 0, 0] = rho_in[:, 0, 0] + p * rho_in[:, 1, 1]
    rho_out[:, 1, 1] = (1 - p) * rho_in[:, 1, 1]
    sqrt_1_p = math.sqrt(1 - p)
    rho_out[:, 0, 1] = rho_in[:, 0, 1] * sqrt_1_p
    rho_out[:, 1, 0] = rho_in[:, 1, 0] * sqrt_1_p
    
    probs, ensemble = density_matrix_to_ensemble(rho_out)
    
    return probs, ensemble


def mix_diameterz_thermal_ensemble(N, E=1.0, beta0=1.0, sigma=0.2):
    """
    Generates an ensemble of thermal mixed states along the Z-axis.
    The inverse temperature (beta) is sampled from a normal distribution.
    
    Args:
        N (int): Number of states.
        E (float): Energy separation between states.
        beta0 (float): Mean inverse temperature.
        sigma (float): Standard deviation of the inverse temperature.
        
    Returns:
        probs (torch.Tensor): Thermal probabilities computed via Boltzmann distribution.
        diameterz_ensemble (torch.Tensor): Z-basis eigenvectors.
    """
    beta = torch.normal(beta0, sigma, size=(N,), dtype=torch.float64)

    bE = beta * E
    Z = 2 * torch.cosh(bE) # Partition function
    
    p0 = torch.exp(-bE) / Z
    p1 = torch.exp(bE) / Z

    probs = torch.stack((p0, p1), dim=1)

    diameterz_ensemble = torch.zeros((N, 2, 2), dtype=torch.complex128) 
    diameterz_ensemble[:, 0, 0] = 1
    diameterz_ensemble[:, 1, 1] = 1

    return probs, diameterz_ensemble

# =============================================================================
# VISUALIZATION FUNCTIONS (PLOTS)
# =============================================================================

def plot_mix(rho, N_plot):
    """
    Plots N_plot states on a single Bloch sphere.
    
    Args:
        rho (torch.Tensor): Density matrices to visualize of shape (N, 2, 2).
        N_plot (int): Number of states from the batch to plot.
    """
    states_to_plot = rho[:N_plot]
    b = Bloch()
    b.point_color = ['r']

    x, y, z = _bloch_xyz_mix(states_to_plot)
    pnts = [
        x.detach().cpu().numpy(),
        y.detach().cpu().numpy(),
        z.detach().cpu().numpy()
    ]
    b.add_points(pnts)
    b.show()

def plot_diffusion_mix(time_rho, color, N_plot):
    """
    Plots 5 sequential Bloch spheres showing the time evolution of the diffusion process.
    Assumes the 'states' tensor contains the timestep in its 4th dimension.
    
    Args:
        time_rho (torch.Tensor): History of density matrices (N, 2, 2, T+1).
        color (str): Point color character (e.g., 'b' for blue, 'r' for red).
        N_plot (int): Number of states to plot per sphere.
    """
    fig, axes = plt.subplots(1, 5, figsize=(5*6, 6), subplot_kw={'projection': '3d'})
    
    for i, ax in enumerate(axes):
        b = Bloch(axes=ax)
        b.point_color = [color]

        # Extracts states at standard intervals (0, 5, 10, 15, 20 assuming T=20)
        x, y, z = _bloch_xyz_mix(time_rho[:N_plot, :, :, i*5])

        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()

        pnts = [x_np, y_np, z_np]
        b.add_points(pnts)
        b.render()

    plt.tight_layout()
    plt.show()

def plot_loss(loss_hist, best_loss_list, steps, logscale=False):
    """
    Plots the evolution of the Loss function throughout the training process.
    
    Args:
        loss_hist (list): List containing the complete loss history for each internal step.
        best_loss_list (list): List with the final accepted loss at the end of each timestep.
        steps (int): Number of optimization steps per timestep t.
        logscale (bool): If True, sets the Y-axis to a logarithmic scale.
    """
    plt.figure(figsize=(8,5))
    plt.plot(loss_hist, color='blue', lw=2, label='Batch Loss')
    plt.plot(np.arange(1, len(best_loss_list)+1) * steps, best_loss_list, 
             marker='o', linestyle='--', color='red', lw=2, label='Best Loss per t')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss vs Step")
    plt.grid(True)
    plt.legend()
    
    if logscale:
        plt.yscale('log')
    plt.show()
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

def _density_matrix_to_ensemble(matrix):
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

def _ensemble_to_density_matrix(probs, ensemble):
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

def _time_ensemble_to_time_matrices(time_probs, time_ensemble):
    """
    Converts the entire time history of diffusion ensembles into density matrices.
    
    Args:
        time_probs (torch.Tensor): History of probabilities of shape (N, 2**n, T+1).
        time_ensemble (torch.Tensor): History of eigenvectors of shape (N, 2**n, 2**n, T+1).
        
    Returns:
        time_matrices (torch.Tensor): History of density matrices of shape (N, 2**n, 2**n, T+1).
    """
    N_total=time_probs.shape[0]
    n=int(math.log2(time_probs.shape[1]))
    T=time_probs.shape[2]-1
    time_matrices = torch.zeros([N_total, 2**n, 2**n, T+1], dtype=torch.complex128)
    for t in range(T+1):
        time_matrices[:,:,:,t] = _ensemble_to_density_matrix(time_probs[:,:,t], time_ensemble[:,:,:,t])
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

def save_simulation(folder_name, filename, N, L, steps, best_params, best_probs, 
                    best_ensemble, loss_dict, ancilla_rand_state, norm):
    """
    Saves all simulation tensors and metadata into a single PyTorch dictionary file (.pt).
    
    Args:
        folder_name (str): Destination directory path.
        filename (str): Name of the file (e.g., 'simulation_L2_N500_ANILLO_dep.pt').
        N, L, steps (int): Simulation hyperparameters.
        best_params, best_probs, best_ensemble, ancilla_rand_state (torch.Tensor): Quantum state tensors.
        loss_dict (dict): Dictionary containing the training metrics (loss, best, wass, kl).
        norm (float): Normalization factor.
    """
    os.makedirs(folder_name, exist_ok=True) 
    file_path = os.path.join(folder_name, filename)
    
    data_dict = {
        'N': N,
        'L': L,
        'steps': steps,
        'best_params': best_params,
        'best_probs': best_probs,
        'best_ensemble': best_ensemble,
        'loss_dict': loss_dict,
        'ancilla_rand_state': ancilla_rand_state,
        'norm': norm
    }
    
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

def _generate_Haar_distribution(N, n):
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

def _generate_maximally_mixed_ensemble(N, n):
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
    states = torch.zeros([N, 2], dtype=torch.complex128) 
    phis = torch.rand(N)*2*torch.pi
    if A == 0:
        r2 = math.sqrt(2.0)
        states[:, 0] = 1/r2
        states[:, 1] = torch.exp(-1j*phis)/r2
    elif A == 1:
        states[:, 0] = torch.cos(phis)
        states[:, 1] = torch.sin(phis)
    elif A == 2:
        states[:, 0] = torch.cos(phis)
        states[:, 1] = -1j * torch.sin(phis)

    rho_pure = torch.einsum('ni,nj->nij', states, states.conj())
    I = torch.eye(2, dtype=torch.complex128).unsqueeze(0).repeat(N, 1, 1)
    
    circle_mix = (1 - 4*p/3) * rho_pure + 4*p/3 * (I/2) 

    probs, ensemble = _density_matrix_to_ensemble(circle_mix)
    return probs, ensemble

def generate_mix_diameter(N, A):
    """
    Generates N mixed states randomly distributed along a specific diameter of the Bloch sphere.
    
    Args:
        N (int): Number of states to generate.
        A (int): Axis of the diameter (0=Z, 1=Y, 2=X).
        
    Returns:
        probs (torch.Tensor): Eigenvalues of shape (N, 2).
        ensemble (torch.Tensor): Eigenvectors of shape (N, 2, 2).
    """

    p0 = torch.rand(N)
    probs = torch.zeros([N, 2], dtype=torch.float64)
    probs[:, 0] = p0
    probs[:, 1] = 1.0 - p0
    
    ensemble = torch.zeros((N, 2, 2), dtype=torch.complex128)
    r2 = math.sqrt(2.0)
    if A == 0: 
        ensemble[:, 0, 0] = 1.0
        ensemble[:, 1, 1] = 1.0
    elif A == 1:
        ensemble[:, 0, 0] = 1.0/r2
        ensemble[:, 1, 0] = 1j/r2
        
        ensemble[:, 0, 1] = 1.0/r2
        ensemble[:, 1, 1] = -1j/r2
    elif A == 2:
        ensemble[:, 0, 0] = 1.0/r2
        ensemble[:, 1, 0] = 1.0/r2
        
        ensemble[:, 0, 1] = 1.0/r2
        ensemble[:, 1, 1] = -1.0/r2

    return probs, ensemble


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
    initial_states=_generate_Haar_distribution(N, 1)

    rho_in = torch.einsum('ni,nj->nij', initial_states, initial_states.conj())
    rho_out = torch.zeros_like(rho_in)
    
    rho_out[:, 0, 0] = rho_in[:, 0, 0]+p*rho_in[:, 1, 1]
    rho_out[:, 1, 1] = (1 - p)*rho_in[:, 1, 1]
    sqrt_1_p = math.sqrt(1 - p)
    rho_out[:, 0, 1] = rho_in[:, 0, 1]*sqrt_1_p
    rho_out[:, 1, 0] = rho_in[:, 1, 0]*sqrt_1_p
    
    probs, ensemble = _density_matrix_to_ensemble(rho_out)
    
    return probs, ensemble


def diameterz_thermal_ensemble(N, E=1.0, beta0=1.0, sigma=0.2):
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
    
    p0 = torch.exp(-bE)/Z
    p1 = torch.exp(bE)/Z

    probs = torch.stack((p0, p1), dim=1)

    diameterz_ensemble = torch.zeros((N, 2, 2), dtype=torch.complex128) 
    diameterz_ensemble[:, 0, 0] = 1
    diameterz_ensemble[:, 1, 1] = 1

    return probs, diameterz_ensemble

# =============================================================================
# VISUALIZATION FUNCTIONS (PLOTS)
# =============================================================================


def plot_bloch_time_distribution(time_probs, time_ensemble, color, N_plot, tsteps=None):
    """
    Plots sequential Bloch spheres showing the time evolution of the diffusion process.
    
    Args:
        time_probs (torch.Tensor): History of probabilities (eigenvalues) of shape (N, 2, T+1).
        time_ensemble (torch.Tensor): History of eigenvectors of shape (N, 2, 2, T+1).
        color (str): Point color character (e.g., 'b' for blue, 'r' for red).
        N_plot (int): Number of states to plot per bloch sphere.
        tsteps (list or 1D array, optional): Specific timestep indices to plot.
                                             Defaults to [0, 5, 10, 15, 20].
    """
    time_rho=_time_ensemble_to_time_matrices(time_probs,time_ensemble)
    if tsteps is None:
        tsteps = [0, 5, 10, 15, 20]
        
    num_spheres = len(tsteps)
    
    fig, axes = plt.subplots(1, num_spheres, figsize=(num_spheres*6, 6), subplot_kw={'projection': '3d'})
    
    if num_spheres == 1:
        axes = [axes]
        
    for ax, t in zip(axes, tsteps):
        b = Bloch(axes=ax)
        b.point_color = [color]

        x, y, z = _bloch_xyz_mix(time_rho[:N_plot, :, :, t])

        pnts = [
            x.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            z.detach().cpu().numpy()
        ]

        b.add_points(pnts)
        b.render()
        ax.set_title(f"t={t}", fontsize=25)

    plt.tight_layout()
    plt.show()



def plot_loss(loss_dict, steps, logscale=False):
    """
    Plots the evolution of the Loss function throughout the training process.
    
    Args:
        loss_dict (dict): Dictionary containing the complete loss history ('loss'), 
                          best loss per timestep ('best'), and the individual components 
                          ('wass', 'kl').
        steps (int): Number of optimization steps per timestep t.
        logscale (bool): If True, sets the Y-axis to a logarithmic scale. Defaults to False.
    """
    loss_hist = loss_dict['loss']
    best_loss_list = loss_dict['best']
    wass_hist = loss_dict['wass']
    kl_hist = loss_dict['kl']

    if sum(kl_hist)>0:
        plt.plot(loss_hist, color='blue', lw=2, label='Total Loss')
        plt.plot(wass_hist, color='orange', lw=1.5, alpha=0.7, label='Wasserstein Component')
        plt.plot(kl_hist, color='green', lw=1.5, alpha=0.7, label='KL Divergence Component')
    else:
        plt.plot(loss_hist, color='blue', lw=2, label='Loss (Wasserstein)')
    plt.plot(np.arange(1, len(best_loss_list)+1)*steps-1, best_loss_list, 
             marker='o', linestyle='--', color='red', lw=2, label='Best Loss per t')
             
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss vs Step")
    plt.grid(True)
    plt.legend()
    if logscale:
        plt.yscale('log')
    plt.show()

def plot_diameter_distribution(probs1, beta0, E, probs2=None, bins=50, color1='r', color2='b', label1='Initial', label2='Denoised'):
    """
    Plots the probability density histogram of a thermal ensemble along the Bloch sphere diameter.
    Dynamically supports plotting either a single distribution or comparing two distributions using perfectly aligned bins.
    
    Args:
        probs1 (torch.Tensor): Primary probabilities (eigenvalues). Shape: [N, 2]
        beta0 (float): Mean inverse temperature of the theoretical distribution.
        E (float): Energy separation of the two-level system.
        probs2 (torch.Tensor, optional): Secondary probabilities for comparison. Shape: [N, 2]
        bins (int, optional): Number of bins for the histogram. Defaults to 50.
        color1 (str, optional): Color for the primary histogram. Defaults to red ('r').
        color2 (str, optional): Color for the secondary histogram. Defaults to blue ('b').
        label1 (str, optional): Legend label for the primary histogram.
        label2 (str, optional): Legend label for the secondary histogram.
    """
    r1 = 2*probs1[:, 0]-1  # p0 - p1
    r1 = r1.detach().cpu().numpy()

    if probs2 is not None:
        r2 = 2*probs2[:, 0]-1
        r2 = r2.detach().cpu().numpy()

        min_val = min(r1.min(), r2.min())
        max_val = max(r1.max(), r2.max())
    else:
        min_val = r1.min()
        max_val = r1.max()
    common_bins = np.linspace(min_val, max_val, bins+1)

    plt.figure()
    plt.hist(r1, bins=common_bins, density=True, alpha=0.5, color=color1, label=label1)
    if probs2 is not None:
        plt.hist(r2, bins=common_bins, density=True, alpha=0.5, color=color2, label=label2)
    plt.axvline(-np.tanh(beta0*E), linestyle="--", color='black', label='Mean')
    plt.xlabel("r (Z-axis coordinate)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.show()

def plot_bloch_classes(probs, ensemble, num_classes, color, N, N_plot):
    """
    Plots multiple Bloch spheres side-by-side to visualize different classes of mixed states.
    Extracts a subset of states from a concatenated batch.
    
    Args:
        probs (torch.Tensor): Concatenated eigenvalues of all classes. Shape: [num_classes*N, 2**n]
        ensemble (torch.Tensor): Concatenated eigenvectors of all classes. Shape: [num_classes*N, 2**n, 2**n]
        num_classes (int): Total number of distinct classes/conditions.
        color (str): Point color character (e.g., 'b' for blue, 'r' for red).
        N (int): Total number of states per class within the concatenated tensors.
        N_plot (int): Number of states to actually plot per class (to prevent visual clutter).
    """
    rhos=_time_ensemble_to_time_matrices(probs,ensemble)
    fig, axes = plt.subplots(1, num_classes, figsize=(num_classes * 6, 6), subplot_kw={'projection': '3d'})
 
    if num_classes == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        b = Bloch(axes=ax)
        b.point_color = [color]

        x, y, z = _bloch_xyz_mix(rhos[i*N : (i*N) + N_plot])

        pnts = [
            x.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            z.detach().cpu().numpy()
        ]

        b.add_points(pnts)
        b.render() 
        ax.set_title(f"Class {i}", fontsize=25)

    plt.tight_layout()
    plt.show()
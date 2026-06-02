import pennylane as qml
import matplotlib.pyplot as plt
from pennylane import numpy as np
from qutip import Bloch
import torch
import math
from qiskit.visualization import plot_state_qsphere


def mix_diameterz_ensemble(N):
    p0=torch.rand(N)
    p1=1-p0

    probs=torch.zeros([N,2],dtype=torch.float64)
    diameterz_ensemble = torch.zeros((N,2,2), dtype=torch.complex128) #numero,state,prob

    probs[:,0]=p0
    probs[:,1]=p1
    diameterz_ensemble[:,0,0]=1
    diameterz_ensemble[:,1,1]=1

    return probs,diameterz_ensemble

def mix_diametery_ensemble(N):
    p0=torch.rand(N)

    probs=torch.zeros([N,2],dtype=torch.float64)
    diametery=torch.zeros((N, 2, 2), dtype=torch.complex128)
    probs[:,0]=p0
    probs[:,1]=1-p0

    r=np.sqrt(2)
    diametery[:,0,0]=1/r
    diametery[:,1,0]=1j/r
    
    diametery[:,0,1]=1/r
    diametery[:,1,1]=-1j/r
    
    return probs,diametery

def circleX(N): 
    states=torch.zeros([N,2], dtype=torch.complex128) 
    phis = torch.rand(N)*2*torch.pi

    states[:, 0] = torch.cos(phis)
    states[:, 1] = -1j*torch.sin(phis)

    return states

def circleY(N): 
    states=torch.zeros([N,2], dtype=torch.complex128) 
    phis = torch.rand(N)*2*torch.pi

    states[:, 0] = torch.cos(phis)
    states[:, 1] = torch.sin(phis)

    return states

def circleZ(N): 
    states=torch.zeros([N,2], dtype=torch.complex128) 
    phis = torch.rand(N)*2*torch.pi
    r2 = math.sqrt(2.0)

    states[:, 0] = 1/r2
    states[:, 1] = torch.exp(-1j*phis)/r2

    return states

def generate_mix_ring(N,p,A):
    if A==0:
        circle_pure = circleZ(N)
    elif A==1:
        circle_pure = circleY(N)
    elif A==2:
        circle_pure = circleX(N)

    rho_pure = torch.einsum('ni,nj->nij', circle_pure, circle_pure.conj())
    
    I = torch.eye(2, dtype=torch.complex128).unsqueeze(0).repeat(N, 1, 1)
    circle_mix=(1-4*p/3)*rho_pure+4*p/3*(I/2) #depolarizing channel

    probs,ensemble=density_matrix_to_ensemble(circle_mix)
    
    return probs,ensemble
    
def ensemble_to_density_matrix(probs, ensemble):
    probs_complex = probs.to(ensemble.dtype) #paso a complejo para multiplicar
    rho = torch.einsum('nk, nik, njk -> nij', probs_complex, ensemble, ensemble.conj())
    
    return rho

def bloch_xyz_mix(states):
    x = states[:,0,1]+states[:,1,0]
    y = 1j*(states[:,0,1]-states[:,1,0])
    z = states[:,0,0]-states[:,1,1]
    return x,y,z

def plot_mix(states,N_plot):
    states_to_plot = states[:N_plot]
    states_to_plot = states_to_plot
    b = Bloch()
    b.point_color = ['r']

    x,y,z=bloch_xyz_mix(states_to_plot)
    #pnts = [x, y, z]
    pnts = [
    x.detach().cpu().numpy(),
    y.detach().cpu().numpy(),
    z.detach().cpu().numpy()
    ]
    b.add_points(pnts)
    b.show()

def plot_diffusion_mix(states, color, N_plot):
    fig, axes = plt.subplots(1, 5, figsize=(5*6, 6), subplot_kw={'projection': '3d'})
    
    for i, ax in enumerate(axes):
        b = Bloch(axes=ax)
        b.point_color = [color]

        x, y, z = bloch_xyz_mix(states[:N_plot,:,:,i*5])

        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()

        pnts = [x_np, y_np, z_np]

        b.add_points(pnts)

        b.render()

    plt.tight_layout()
    plt.show()

def density_matrix_to_ensemble(matrix):
    probs, ensemble = torch.linalg.eigh(matrix)
    return probs,ensemble

def time_ensemble_to_time_matrices(N,n,T,time_probs,time_ensemble):
    time_matrices=torch.zeros([N,2**n,2**n,T+1], dtype=torch.complex128)
    for t in range(T+1):
        time_matrices[:,:,:,t]=ensemble_to_density_matrix(time_probs[:,:,t], time_ensemble[:,:,:,t])
    return time_matrices

def generate_Haar_distribution(N,n):
    states=torch.randn(N, 2**n, dtype=torch.float64) + 1j*torch.randn(N, 2**n, dtype=torch.float64)
    norms=torch.linalg.norm(states, dim=1, keepdim=True)
    haar_states=states/norms
    return haar_states

def plot_loss(loss_hist,best_loss_list, steps, logscale=False):
    plt.figure(figsize=(8,5))
    plt.plot(loss_hist, color='blue', lw=2)
    plt.plot(np.arange(1,len(best_loss_list)+1)*steps,best_loss_list, marker='o', linestyle='--', color='red', lw=2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss vs Step")
    plt.grid(True)
    if logscale:
        plt.yscale('log')
    plt.show()

def generate_maximally_mixed_ensemble(N, n):
    """
    Genera un batch de N estados máximamente mixtos para n qubits.
    Devuelve los tensores en formato (probs, ensemble) listos para la inferencia.
    """

    # 1. Probabilidades (Autovalores): Distribución uniforme
    probs = torch.ones([N, 2**n], dtype=torch.float64) / 2**n
    
    # 2. Ensambles (Autovectores): Matriz Identidad repetida N veces
    I = torch.eye(2**n, dtype=torch.complex128)
    ensemble = I.unsqueeze(0).repeat(N, 1, 1)
    
    return probs, ensemble
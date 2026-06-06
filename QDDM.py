import pennylane as qml
import torch
from utiles import _density_matrix_to_ensemble, _generate_Haar_distribution, _generate_maximally_mixed_ensemble
import ot
import math


class QuantumDiffusion():
    """
    Simulates the forward diffusion process by incrementally adding quantum noise 
    (either through depolarizing channels or scrambling circuits) to a batch of initial states.
    """
    def __init__(self,num_qubits:int,delta_t:torch.Tensor=None,p_t:torch.Tensor=None): #no poner numero de estados para poder variarlo
        """
        Initializes the Quantum Diffusion environment.
        
        Args:
            num_qubits (int): Number of qubits in the main system.
            delta_t (torch.Tensor, optional): Time steps for the scrambling process. Shape (T)
            p_t (torch.Tensor, optional): Noise schedule (probabilities) for the depolarizing process. Shape (T)
        """
        self.n=num_qubits
        self.delta_t=delta_t #pasar delta_t y p_t?
        self.p_t=p_t
        if (delta_t is not None):
            self.T=len(delta_t)
        else:
            self.T=len(p_t)

        self.dev_scr = qml.device("default.qubit", wires=self.n)
        self.dev_dep = qml.device("default.mixed", wires=self.n)

        self.dep_diffusion=qml.QNode(self._dep_circuit,self.dev_dep,interface="torch")
        self.scr_diffusion=qml.QNode(self._scr_circuit,self.dev_scr,interface="torch")

    def _dep_circuit(self,states,p):
        """
        Applies a local depolarizing channel to each qubit.
        
        Args:
            states (torch.Tensor): Input states to be noised. Shape (N, 2**n)
            p (float): Depolarizing probability for the current timestep.
            
        Returns:
            torch.Tensor: Density matrix of the noised state. Shape (N, 2**n, 2**n)
        """
        qml.StatePrep(states, wires=range(self.n)) #N circuitos en paralelo
        for i in range(self.n): #esto seria un depolarizing local, se podria hacer global pero ahora solo hay 1 qubit
            qml.DepolarizingChannel(p, wires=i)
        return qml.density_matrix(wires=range(self.n))

    def _scr_circuit(self,omegas,omegas2,states,t):
        """
        Applies a parameterized scrambling circuit using rotations and IsingZZ couplings.
        
        Args:
            omegas (torch.Tensor): Single-qubit rotations. Shape: [T, n, 3, N]
            omegas2 (torch.Tensor): IsingZZ couplings. Shape: [T, n, N]
            states (torch.Tensor): Input pure states. Shape: [N, 2**n]
            t (int): Current timestep index.
            
        Returns:
            torch.Tensor: Scrambled pure states. Shape: [N, 2**n]
        """
        qml.StatePrep(states, wires=range(self.n)) #N circuitos en paralelo
        for i in range(self.n): #qubits
            qml.RZ(omegas[t,i,0],wires=i)
            qml.RY(omegas[t,i,1],wires=i)
            qml.RZ(omegas[t,i,2],wires=i)
        if self.n == 2:
            qml.IsingZZ(omegas2[t,0], wires=[0, 1])
        elif (self.n>2):
            for i in range(self.n):
                qml.IsingZZ(omegas2[t, i],wires=[i,(i+1)%self.n])
        return qml.state()
    
    def forward_scr(self,N:int,probs:torch.Tensor,ensemble:torch.Tensor):
        """
        Executes the forward diffusion process over all timesteps T using scrambling circuits.
        
        Args:
            N (int): Number of states in the batch.
            probs (torch.Tensor): Initial probabilities. Shape: [N, 2**n]
            ensemble (torch.Tensor): Initial states ensemble. Shape: [N, 2**n, 2**n]
            
        Returns:
            torch.Tensor: Time history of the ensemble. Shape: [N, 2**n, 2**n, T+1]
        """
        omegas = torch.rand(self.T, self.n, 3, N)*(torch.pi/4) - (torch.pi/8)
        omegas = omegas * self.delta_t[:,None,None,None]
        omegas2= torch.rand(self.T, self.n, N)*(torch.pi/4) - (torch.pi/8) #para n=1,2 sobra un parametro pero da un poco igual
        omegas2 = omegas2 * self.delta_t[:,None,None]

        time_ensemble=torch.zeros([N,2**self.n,2**self.n,self.T+1], dtype=torch.complex128) #number,state,prob,time
        time_ensemble[:,:,:,0]=ensemble
        time_probs=torch.zeros([N,2**self.n,self.T+1], dtype=torch.float64)
        time_probs[:,:,0]=probs
        for t in range(self.T):
            time_ensemble[:,:,0,t+1]=self.scr_diffusion(omegas,omegas2,time_ensemble[:,:,0,t],t)
            time_ensemble[:,:,1,t+1]=self.scr_diffusion(omegas,omegas2,time_ensemble[:,:,1,t],t)
            time_probs[:,:,t+1]=probs
        return time_probs,time_ensemble

    def forward_dep(self,N:int,probs:torch.Tensor,ensemble):
        """
        Executes the forward diffusion process over all timesteps T using depolarizing noise.
        
        Args:
            N (int): Number of states in the batch.
            probs (torch.Tensor): Initial probabilities. Shape: [N, 2**n]
            ensemble (torch.Tensor): Initial eigenvectors. Shape: [N, 2**n, 2**n]
            
        Returns:
            time_probs (torch.Tensor): Probability history. Shape: [N, 2**n, T+1]
            time_ensemble (torch.Tensor): Ensemble history. Shape: [N, 2**n, 2**n, T+1]
        """
        time_ensemble=torch.zeros([N,2**self.n,2**self.n,self.T+1], dtype=torch.complex128) #numero,state,prob,tiempo
        time_probs=torch.zeros([N,2**self.n,self.T+1], dtype=torch.float64)
        time_ensemble[:,:,:,0]=ensemble
        time_probs[:,:,0]=probs
        for t in range(self.T):
            total_matrix=torch.zeros([N,2**self.n,2**self.n],dtype=torch.complex128)
            for i in range(2**self.n):
                output_matrix = self.dep_diffusion(time_ensemble[:,:,i,t],self.p_t[t])
                total_matrix=total_matrix+time_probs[:,i,t].view(-1, 1, 1)*output_matrix
            time_probs[:,:,t+1], time_ensemble[:,:,:,t+1]=_density_matrix_to_ensemble(total_matrix)
        return time_probs,time_ensemble
        


class QuantumDenoising():
    """
    Implements the backward denoising process using a Parameterized Quantum Circuit (PQC).
    Learns to reverse the quantum noise step-by-step using Wasserstein distance optimization.
    """
    def __init__(self, n:int, n_a:int, n_c:int, T:int, L: int, num_classes:int=1): 
        """
        Initializes the Quantum Denoising model.
        
        Args:
            n (int): Number of main system qubits.
            n_a (int): Number of haar distribution ancilla qubits.
            n_c (int): Number of zero state ancilla qubits. Can be used for conditioning.
            T (int): Total number of diffusion timesteps.
            L (int): Number of parameterized layers in the PQC.
            num_classes (int): Number of classes for conditioning.
        """
        self.n=n
        self.n_a=n_a
        self.n_c=n_c
        self.n_total=n+n_a+n_c
        self.T=T
        self.L=L
        self.num_classes=num_classes

        self._sqrt_fun = torch.sqrt

        self.dev_den = qml.device("default.qubit", wires=self.n_total)
        self.PQC_denoising=qml.QNode(self._PQC,self.dev_den,interface="torch",diff_method="backprop")

    def _sqrt_strategy(self, sqrt=True):
        """
        Dynamically sets the cost transformation strategy.
        If True, applies the square root to the core cost matrix (HS distance) before calculating 
        the Wasserstein distance. If False, uses the squared cost.
        """
        self._sqrt_fun = torch.sqrt if sqrt else lambda x: x

    def _PQC(self, train_state, thetas, mu):
        """
        Parameterized Quantum Circuit serving as the denoising network.
        
        Args:
            train_state (torch.Tensor): Input noisy state + ancillas + cond. Shape: [N_total, 2**n_total]
            thetas (torch.Tensor): Trainable parameters of the circuit. Shape: [L, n_total, 2]
            mu (torch.Tensor): Conditioning angle parameter. Shape: [N_total]
            
        Returns:
            torch.Tensor: Reduced density matrix of the main system. Shape: [N_total, 2**n, 2**n]
        """
        qml.StatePrep(train_state, wires=range(self.n_total))
        for i in range(self.n+self.n_a, self.n_total):
            qml.RX(mu, wires=i)

        for l in range(self.L):
            for i in range(self.n_total):
                qml.RX(thetas[l, i, 0], wires=i) 
                qml.RY(thetas[l, i, 1], wires=i)

            for i in range(0, self.n_total-1, 2):
                qml.CZ(wires=[i, i+1])
            for i in range(1, self.n_total-1, 2):
                qml.CZ(wires=[i, i+1])

        return qml.density_matrix(wires=range(self.n))
    
    def _wasserstein_distance(self, N, true_ensemble_class, true_probs_class, gen_ensemble_class, gen_probs_class):
        """
        Calculates the Wasserstein distance between true and generated ensembles using
        the Hilbert_Schmidt distance between all combinations of true and generated mix states.
        
        Args:
            N (int): Batch size per class.
            true_ensemble_class (torch.Tensor): Target eigenvectors. Shape: [N, 2**n, 2**n]
            true_probs_class (torch.Tensor): Target eigenvalues. Shape: [N, 2**n]
            gen_ensemble_class (torch.Tensor): Generated eigenvectors. Shape: [N, 2**n, 2**n]
            gen_probs_class (torch.Tensor): Generated eigenvalues. Shape: [N, 2**n]
            
        Returns:
            torch.Tensor: Scalar Wasserstein distance. Shape: []
        """
        inner_prod_true = torch.einsum('ndi, ndj -> nij', torch.conj(true_ensemble_class), true_ensemble_class)
        overlaps_pure_true = torch.abs(inner_prod_true)**2
        purity_true = torch.einsum('ni, nj, nij -> n', true_probs_class, true_probs_class, overlaps_pure_true)

        inner_prod_gen = torch.einsum('mdi, mdj -> mij', torch.conj(gen_ensemble_class), gen_ensemble_class)
        overlaps_pure_gen = torch.abs(inner_prod_gen)**2
        purity_gen = torch.einsum('mi, mj, mij -> m', gen_probs_class, gen_probs_class, overlaps_pure_gen)

        inner_prod_cross = torch.einsum('ndi, mdj -> nmij', torch.conj(true_ensemble_class), gen_ensemble_class)
        overlaps_pure_cross = torch.abs(inner_prod_cross)**2
        overlaps_cross = torch.einsum('ni, mj, nmij -> nm', true_probs_class, gen_probs_class, overlaps_pure_cross)

        Cij = self._sqrt_fun(purity_true.unsqueeze(1) + purity_gen.unsqueeze(0) - 2.0 * overlaps_cross)
        M_cost = Cij.to(torch.float64)

        a = torch.ones(N, dtype=torch.float64)/N
        Wd = ot.emd2(a, a, M_cost) 

        return Wd
        
    def _wasserstein_loss(self, N, true_ensemble, true_probs, gen_ensemble, gen_probs, norm):
        """
        Aggregates the Wasserstein distance across all condition classes and applies normalization.
        
        Args:
            N (int): Batch size per class.
            true_ensemble (torch.Tensor): Full target eigenvectors. Shape: [N_total, 2**n, 2**n]
            true_probs (torch.Tensor): Full target eigenvalues. Shape: [N_total, 2**n]
            gen_ensemble (torch.Tensor): Full generated eigenvectors. Shape: [N_total, 2**n, 2**n]
            gen_probs (torch.Tensor): Full generated eigenvalues. Shape: [N_total, 2**n]
            norm (float): Normalization constant.
            
        Returns:
            torch.Tensor: Normalized total loss. Shape: []
        """
        total = torch.tensor(0.0, dtype=torch.float64)
        for i in range(self.num_classes):
            true_ensemble_class = true_ensemble[i*N:(i+1)*N]
            true_probs_class = true_probs[i*N:(i+1)*N]
            gen_ensemble_class = gen_ensemble[i*N:(i+1)*N]
            gen_probs_class = gen_probs[i*N:(i+1)*N]

            Wd=self._wasserstein_distance(N, true_ensemble_class, true_probs_class, gen_ensemble_class, gen_probs_class)
            total = total + Wd 

        return (total/self.num_classes)/norm 


    def _norm_loss(self, N, noisy_ensemble, noisy_probs, true_ensemble_in, true_probs_in): ###esto igual habria que revalorarlo
        """
        Computes the normalization factor based on the distance between clean and fully noisy states.
        Takes the greatest distance of all the classes
        
        Args:
            N (int): Batch size per class.
            noisy_ensemble, noisy_probs (torch.Tensor): States at t=T. Shapes: [N_total, 2**n, 2**n], [N_total, 2**n]
            true_ensemble_in, true_probs_in (torch.Tensor): States at t=0. Shapes: [N_total, 2**n, 2**n], [N_total, 2**n]
            
        Returns:
            float: Maximum Wasserstein distance across classes.
        """
        norm = -1 * float('inf')
        for i in range(self.num_classes):
            true_ensemble_class = true_ensemble_in[i*N:(i+1)*N]
            true_probs_class = true_probs_in[i*N:(i+1)*N]
            gen_ensemble_class = noisy_ensemble[i*N:(i+1)*N]
            gen_probs_class = noisy_probs[i*N:(i+1)*N]

            Wd=self._wasserstein_distance(N, true_ensemble_class, true_probs_class, gen_ensemble_class, gen_probs_class)
            if norm < Wd:
                norm = Wd
        return norm
    
    def _KL_divergence_loss(self, true_probs,gen_probs):
        grid = torch.linspace(-1, 1, steps=50, dtype=torch.float64)

        r1 = 2*true_probs[:,0]-1
        r2 = 2*gen_probs[:,0]-1

        diff_true = r1.unsqueeze(1) - grid.unsqueeze(0)
        diff_gen = r2.unsqueeze(1) - grid.unsqueeze(0)

        #Gaussian (e^(-0.5*(x-mu)/sigma)^2))
        pdf_true = torch.exp(-0.5*(diff_true/0.1)**2)
        pdf_gen = torch.exp(-0.5*(diff_gen/0.1)**2)

        #sum of N gaussians y every grid point
        p_true = pdf_true.sum(dim=0)
        p_true = p_true/p_true.sum()

        p_gen = pdf_gen.sum(dim=0)
        p_gen = p_gen/p_gen.sum()

        p_true = p_true+1e-10
        p_gen = p_gen+1e-10
        
        p_true = p_true/p_true.sum()
        p_gen = p_gen/p_gen.sum()

        kl_loss = torch.sum(p_true*torch.log(p_true/p_gen))
        
        return kl_loss
    
    def _sample_states_from_circuit(self,N_total,thetas, states_in,probs, mu, ancilla_state,cond_state):
        """
        Passes a batch of ensembles through the denoising PQC and extracts th resulting ensemble distribution.
        
        Args:
            N_total (int): Total batch size across all classes (N * num_classes).
            thetas (torch.Tensor): Current parameters of the PQC. Shape: [L, n_total, 2]
            states_in (torch.Tensor): Input noise eigenvectors. Shape: [N_total, 2**n, 2**n]
            probs (torch.Tensor): Input noise eigenvalues. Shape: [N_total, 2**n]
            mu (torch.Tensor): Conditioning parameters. Shape: [N_total]
            ancilla_state (torch.Tensor): Haar ancilla states. Shape: [N_total, 2**n_a] (or [N_total, 1] if n_a=0)
            cond_state (torch.Tensor): Zero ancilla states (useed for conditioning). Shape: [2**n_c]
            
        Returns:
            probs_new (torch.Tensor): Denoised eigenvalues. Shape: [N_total, 2**n]
            ensemble_new (torch.Tensor): Denoised eigenvectors. Shape: [N_total, 2**n, 2**n]
        """
        #producto kroneker sistema ancilla
        rand_states = torch.einsum('bij,bk->bikj', states_in, ancilla_state).reshape(N_total, 2**(self.n+self.n_a),2**(self.n))
        #producto de kroneker sistema ancilla cond
        input_states=torch.einsum('bij,k->bikj', rand_states, cond_state).reshape(N_total, 2**self.n_total, 2**self.n)

        total_matrix=torch.zeros([N_total,2**self.n,2**self.n],dtype=torch.complex128)
        for i in range(2**self.n):
            output_matrix = self.PQC_denoising(input_states[:,:,i], thetas, mu)
            total_matrix=total_matrix+probs[:,i].view(-1, 1, 1)*output_matrix
            
        probs_new, ensemble_new = torch.linalg.eigh(total_matrix) #autovalores (N,2), autovectores (N,2,2) en columnas:  autovects[i, :, 0] corresponde a autovals[i, 0]

        return probs_new, ensemble_new
        
    def backward_denoising(self,N,time_probs,time_ensemble,steps,Haar_start=False,kl_lambda=0.0,sqrt=True,T_start=None,
                           ancilla_state_in=None,norm_in=None,prev_params=None, prev_probs=None,prev_ensemble=None,prev_history=None):
        """
        Main training loop for the backward denoising process.
        Optimizes the PQC parameters step-by-step from t=T back to t=0.
        
        Args:
            N (int): Batch size per class.
            time_probs (torch.Tensor): Target eigenvalues history. Shape: [N_total, 2**n, T+1]
            time_ensemble (torch.Tensor): Target eigenvectors history. Shape: [N_total, 2**n, 2**n, T+1]
            steps (int): Number of optimization steps per timestep.
            Haar_start (bool, optional): In case of a scrambling diffusion, you can start from a Haar distribution as starting point. 
            kl_lambda (float, optional): Weight of the KL divergence penalty in the total loss. Defaults to 0.0.
            sqrt (bool, optional): If True, applies square root to the core cost matrix in Wasserstein calculation
                                   (HS distance). Defaults to True.

            --- Training Checkpoint Tools (Optional tools for redoing failed training from T_start) ---
            T_start (int, optional): The exact timestep to resume training from. If None, starts from t=T.
            ancilla_state_in (torch.Tensor, optional): The saved Haar random states used for ancillas. Required to maintain consistency. 
                                                       Shape: [N_total, 2**n_a]
            norm_in (float, optional): The saved normalization factor (max Wasserstein distance). 
            prev_params (torch.Tensor, optional): The network weights (thetas) learned up in the previous simulation. 
                                                  Shape: [T, L, n_total, 2]
            prev_probs (torch.Tensor, optional): The history of reconstructed eigenvalues from the previous simulation. 
                                                 Shape: [N_total, 2**n, T+1]
            prev_ensemble (torch.Tensor, optional): The history of reconstructed eigenvectors from the previous simulation. 
                                                    Shape: [N_total, 2**n, 2**n, T+1]
            prev_history (dict, optional): Dictionary containing the history of losses ('loss', 'best', 'wass', 'kl').
        
        Returns:
            best_params (torch.Tensor): Trained network weights. Shape: [T, L, n_total, 2]
            best_ensemble_probs (torch.Tensor): Reconstructed eigenvalues. Shape: [N_total, 2**n, T+1]
            best_ensemble (torch.Tensor): Reconstructed eigenvectors. Shape: [N_total, 2**n, 2**n, T+1]
            loss_dict (dict): Dictionary with all loss tracking histories.
            ancilla_rand_state (torch.Tensor): Generated ancilla states (needed if retraining: ancilla_state_in). Shape: [N_total, 2**n_a]
            norm (float): Normalization factor applied (needed if retraining: norm_in).
        """
        salto=(2*math.pi)/self.num_classes
        mu=torch.arange(self.num_classes)*salto
        mu = torch.repeat_interleave(mu, N)

        N_total=time_ensemble.shape[0]

        #ancillas para el conditioning
        cond_state = torch.zeros(2**self.n_c, dtype=torch.complex128)
        cond_state[0] = 1.0

        self._sqrt_strategy(sqrt)

        if kl_lambda>0.0:
            def _compute_loss(N, t_ens, t_pr, g_ens, g_pr, norm, kl_lambda):
                loss_wass = self._wasserstein_loss(N, t_ens, t_pr, g_ens, g_pr, norm)
                loss_kl = self._KL_divergence_loss(t_pr, g_pr)
                total = loss_wass + (kl_lambda*loss_kl)
                return total, loss_wass, loss_kl
        else:
            def _compute_loss(N, t_ens, t_pr, g_ens, g_pr, norm, kl_lambda):
                loss_wass = self._wasserstein_loss(N, t_ens, t_pr, g_ens, g_pr, norm)
                zero_tensor = torch.tensor(0.0)
                return loss_wass, loss_wass, zero_tensor
            

        if T_start is not None:
            ancilla_rand_state = ancilla_state_in
            best_params = prev_params.clone()
            best_ensemble = prev_ensemble.clone()
            best_ensemble_probs = prev_probs.clone()
            norm=norm_in

            ptsteps = self.T-T_start
            cut = ptsteps*steps
            loss_hist = prev_history['loss'][:cut]
            loss_hist_wass = prev_history['wass'][:cut]
            loss_hist_kl = prev_history['kl'][:cut]
            best_loss_list = prev_history['best'][:ptsteps]
        else:
            T_start = self.T

            if self.n_a > 0:
                ancilla_rand_state = _generate_Haar_distribution(N_total, self.n_a) #para Haar de n_a qubits
            else:
                ancilla_rand_state = torch.ones((N_total, 1), dtype=torch.complex128) # Un escalar dummy para que el einsum no haga nada raro

            best_params = torch.zeros([self.T, self.L, self.n_total, 2])
            best_ensemble = torch.zeros([N_total, 2**self.n,2**self.n, self.T+1], dtype=torch.complex128) 
            best_ensemble_probs = torch.zeros([N_total, 2**self.n, self.T+1], dtype=torch.float64) 
            if Haar_start is False:
                best_ensemble[:,:,:,T_start] = time_ensemble[:, :,:, T_start]
                best_ensemble_probs[:,:,T_start]=time_probs[:,:,T_start]
            else:
                Haar_initial=_generate_Haar_distribution(N, self.n)
                Haar_ensemble=torch.zeros([N_total,2**self.n,2**self.n],dtype=torch.complex128)
                Haar_ensemble[:,:,0]=Haar_initial
                probs_Haar=torch.zeros([N_total,2**self.n],dtype=torch.float64)
                probs_Haar[:,0]=1.0
                best_ensemble[:,:,:,T_start] = Haar_ensemble
                best_ensemble_probs[:,:,T_start]=probs_Haar

            norm=self._norm_loss(N,time_ensemble[:,:,:,T_start],time_probs[:,:,T_start],time_ensemble[:,:,:,0],time_probs[:,:,0]) #revisar

            loss_hist = []       
            best_loss_list = []
            loss_hist_wass = []  
            loss_hist_kl = []

        for t in range(T_start-1, -1, -1):  
            print(f"\n========== Denoising t={t} ==========")
            
            train_ensemble = best_ensemble[:, :,:, t+1].detach().clone()
            train_probs=best_ensemble_probs[:, :, t+1].detach().clone()
            target_ensemble = time_ensemble[:, :,:, t].detach().clone()
            target_probs = time_probs[:, :, t].detach().clone()
            
            thetas = torch.rand((self.L, self.n_total, 2), requires_grad=True, dtype=torch.float64)
            optimizer = torch.optim.Adam([thetas], lr=0.01)

            best_loss = float('inf')
            best_current_params = None 
            
            for step in range(1,steps+1):
                optimizer.zero_grad() #reinicia el gradiente del optimizador

                gen_probs,gen_ensemble = self._sample_states_from_circuit(N_total,thetas,train_ensemble,train_probs,mu,ancilla_rand_state,cond_state)
                
                loss,loss_wass,loss_kl = _compute_loss(N, target_ensemble, target_probs, gen_ensemble, gen_probs, norm, kl_lambda)
                
                loss.backward() #obtiene los gradientes
                optimizer.step() 
                
                current_loss = loss.item()
                loss_hist.append(current_loss)
                loss_hist_wass.append(loss_wass.item())  
                loss_hist_kl.append(loss_kl.item())
                
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_current_params = thetas.detach().clone()
                    
                if step % 100 == 0 or step == 1:
                    if kl_lambda > 0.0:
                        print(f"Step {step} | Total Loss: {current_loss:.6f} | Wass: {loss_wass.item():.6f} | KL: {loss_kl.item():.6f}")
                    else:
                        print(f"Step {step} | Loss: {current_loss:.6f}")
            best_loss_list.append(best_loss)
                    
            print(f"-> Best loss: {best_loss:.6f}")
            
            best_params[t, :,:,:] = best_current_params
            
            with torch.no_grad():
                best_gen_probs,best_gen_ensemble=self._sample_states_from_circuit(N_total,best_current_params,train_ensemble,train_probs,mu,ancilla_rand_state,cond_state)
                best_ensemble[:, :,:, t] = best_gen_ensemble
                best_ensemble_probs[:,:,t]=best_gen_probs
            
            loss_dict = {
            "loss": loss_hist,
            "best": best_loss_list,
            "wass": loss_hist_wass,
            "kl": loss_hist_kl
        }

        return best_params,best_ensemble_probs,best_ensemble,loss_dict,ancilla_rand_state,norm
        

    def backward_test(self,N,best_params,Haar_start=False,cls=0): 
        """
        Evaluates the trained denoising model starting from a fully noisy state.
        
        Args:
            N (int): Number of test states per batch.
            best_params (torch.Tensor): Trained network weights. Shape: [T, L, n_total, 2]
            Haar_start (bool, optional): If True, initializes noise from a Haar random distribution (scrambling). 
                                  If False, uses maximally mixed states (depolarizing). Defaults to False.
            cls (int, optional): Class condition to drive the denoising trajectory (e.g. if 3 classes: 0=first, 1=second, 2=third).
                                 By default the first class (0).
            
        Returns:
            time_probs (torch.Tensor): Reconstructed probabilities history. Shape: [N, 2**n, T+1]
            time_ensemble (torch.Tensor): Reconstructed eigenvectors history. Shape: [N, 2**n, 2**n, T+1]
        """
        salto=(2*math.pi)/self.num_classes
        mu_total=(torch.arange(self.num_classes)*salto).to(torch.float64)
        mu=mu_total[cls]

        if self.n_a > 0:
            ancilla_state = _generate_Haar_distribution(N, self.n_a) #para Haar de n_a qubits
        else:
            ancilla_state = torch.ones((N, 1), dtype=torch.complex128)

        cond_state = torch.zeros((2**self.n_c), dtype=torch.complex128)
        cond_state[0] = 1.0

        if Haar_start==False:
            probs_in, ensemble_in=_generate_maximally_mixed_ensemble(N,self.n)
        else:
            haar_initial=_generate_Haar_distribution(N, self.n)
            ensemble_in=torch.zeros([N,2**self.n,2**self.n],dtype=torch.complex128)
            ensemble_in[:,:,0]=haar_initial
            probs_in=torch.zeros([N,2**self.n],dtype=torch.float64)
            probs_in[:,0]=1.0
        
        time_ensemble = torch.zeros([N, 2**self.n,2**self.n, self.T+1], dtype=torch.complex128)
        time_probs = torch.zeros([N,2**self.n, self.T+1], dtype=torch.float64)
        current_ensemble=ensemble_in
        current_probs=probs_in
        time_ensemble[:,:,:,self.T]=current_ensemble
        time_probs[:,:,self.T]=current_probs
        for t in range(self.T-1, -1, -1):
            thetas=best_params[t,:,:,:]

            gen_probs,gen_ensemble=self._sample_states_from_circuit(N,thetas, current_ensemble,current_probs, mu, ancilla_state,cond_state)
            current_ensemble=gen_ensemble
            current_probs=gen_probs
            time_ensemble[:,:,:,t]=current_ensemble
            time_probs[:,:,t]=current_probs
                
        return time_probs,time_ensemble

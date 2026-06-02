import pennylane as qml
import torch
from utiles import density_matrix_to_ensemble, generate_Haar_distribution
import ot
import math


class QuantumDiffusion():
    def __init__(self,num_qubits:int,delta_t:torch.Tensor=None,p_t:torch.Tensor=None): #no poner numero de estados para poder variarlo
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
        qml.StatePrep(states, wires=range(self.n)) #N circuitos en paralelo
        for i in range(self.n): #esto seria un depolarizing local, se podria hacer global pero ahora solo hay 1 qubit
            qml.DepolarizingChannel(p, wires=i)
        return qml.density_matrix(wires=range(self.n))

    def _scr_circuit(self,omegas,omegas2,states,t):
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
    
    def forward_scr(self,N:int,ensemble:torch.Tensor):
        omegas = torch.rand(self.T, self.n, 3, N)*(torch.pi/4) - (torch.pi/8)
        omegas = omegas * self.delta_t[:,None,None,None]
        omegas2= torch.rand(self.T, self.n, N)*(torch.pi/4) - (torch.pi/8) #para n=1,2 sobra un parametro pero da un poco igual
        omegas2 = omegas2 * self.delta_t[:,None,None]

        time_ensemble=torch.zeros([N,2**self.n,2**self.n,self.T+1], dtype=torch.complex128) #numero,state,prob,tiempo
        time_ensemble[:,:,:,0]=ensemble
        for t in range(self.T):
            time_ensemble[:,:,0,t+1]=self.scr_diffusion(omegas,omegas2,time_ensemble[:,:,0,t],t)
            time_ensemble[:,:,1,t+1]=self.scr_diffusion(omegas,omegas2,time_ensemble[:,:,1,t],t)
        return time_ensemble

    def forward_dep(self,N:int,probs,ensemble):
        time_ensemble=torch.zeros([N,2**self.n,2**self.n,self.T+1], dtype=torch.complex128) #numero,state,prob,tiempo
        time_probs=torch.zeros([N,2**self.n,self.T+1], dtype=torch.float64)
        time_ensemble[:,:,:,0]=ensemble
        time_probs[:,:,0]=probs
        for t in range(self.T):
            total_matrix=torch.zeros([N,2**self.n,2**self.n],dtype=torch.complex128)
            for i in range(2**self.n):
                output_matrix = self.dep_diffusion(time_ensemble[:,:,i,t],self.p_t[t])
                total_matrix=total_matrix+time_probs[:,i,t].view(-1, 1, 1)*output_matrix
            time_probs[:,:,t+1], time_ensemble[:,:,:,t+1]=density_matrix_to_ensemble(total_matrix)
        return time_probs,time_ensemble
        


class QuantumDenoising():
    def __init__(self, n:int, n_a:int, n_c:int, T:int, L: int, Class:int):
        self.n=n
        self.n_a=n_a
        self.n_c=n_c
        self.n_total=n+n_a+n_c
        self.T=T
        self.L=L
        self.Class=Class

        self.dev_den = qml.device("default.qubit", wires=self.n_total)
        self.PQC_denoising=qml.QNode(self._PQC,self.dev_den,interface="torch",diff_method="backprop")

    def _PQC(self, train_state, thetas, mu):
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
        inner_prod_true = torch.einsum('ndi, ndj -> nij', torch.conj(true_ensemble_class), true_ensemble_class)
        overlaps_pure_true = torch.abs(inner_prod_true)**2
        purity_true = torch.einsum('ni, nj, nij -> n', true_probs_class, true_probs_class, overlaps_pure_true)

        inner_prod_gen = torch.einsum('mdi, mdj -> mij', torch.conj(gen_ensemble_class), gen_ensemble_class)
        overlaps_pure_gen = torch.abs(inner_prod_gen)**2
        purity_gen = torch.einsum('mi, mj, mij -> m', gen_probs_class, gen_probs_class, overlaps_pure_gen)

        inner_prod_cross = torch.einsum('ndi, mdj -> nmij', torch.conj(true_ensemble_class), gen_ensemble_class)
        overlaps_pure_cross = torch.abs(inner_prod_cross)**2
        overlaps_cross = torch.einsum('ni, mj, nmij -> nm', true_probs_class, gen_probs_class, overlaps_pure_cross)

        Cij = torch.sqrt(purity_true.unsqueeze(1) + purity_gen.unsqueeze(0) - 2.0 * overlaps_cross)
        M_cost = Cij.to(torch.float64)

        a = torch.ones(N, dtype=torch.float64)/N
        Wd = ot.emd2(a, a, M_cost) 

        return Wd
        
    def _wasserstein_loss(self, N, true_ensemble, true_probs, gen_ensemble, gen_probs, norm):
        total = torch.tensor(0.0, dtype=torch.float64)
        for i in range(self.Class):
            true_ensemble_class = true_ensemble[i*N:(i+1)*N]
            true_probs_class = true_probs[i*N:(i+1)*N]
            gen_ensemble_class = gen_ensemble[i*N:(i+1)*N]
            gen_probs_class = gen_probs[i*N:(i+1)*N]

            Wd=self._wasserstein_distance(N, true_ensemble_class, true_probs_class, gen_ensemble_class, gen_probs_class)
            total = total + Wd 

        return (total / self.Class) / norm 


    def _norm_loss(self, N, noisy_ensemble, noisy_probs, true_ensemble_in, true_probs_in):
        norm = -1 * float('inf')
        for i in range(self.Class):
            true_ensemble_class = true_ensemble_in[i*N:(i+1)*N]
            true_probs_class = true_probs_in[i*N:(i+1)*N]
            gen_ensemble_class = noisy_ensemble[i*N:(i+1)*N]
            gen_probs_class = noisy_probs[i*N:(i+1)*N]

            Wd=self._wasserstein_distance(N, true_ensemble_class, true_probs_class, gen_ensemble_class, gen_probs_class)
            if norm < Wd:
                norm = Wd
        return norm
    
    def _sample_states_from_circuit(self,N_total,thetas, states_in,probs, mu, ancilla_state,cond_state):
        #cambio ancilla para cada optimizacion
        #ancilla_rand_state = Haar_distribution_n(N_total, self.n_a)
        #rand_states = torch.einsum('bij,bk->bikj', states_in, ancilla_rand_state).reshape(N_total, 2**(self.n+self.n_a),2**self.n)
        
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
        
    def backward_denoising(self,N,time_ensemble,time_probs,steps,T_start=None,ancilla_state_in=None,norm_in=None,Haar_initial=None,prev_params=None, prev_ensemble=None, prev_probs=None):
        salto=(2*math.pi)/self.Class
        mu=torch.arange(self.Class)*salto

        N_total=time_ensemble.shape[0]

        if T_start is None:
            T_start = self.T
        
        if ancilla_state_in is not None:
            ancilla_rand_state = ancilla_state_in
        else:
            if self.n_a > 0:
                ancilla_rand_state = generate_Haar_distribution(N_total, self.n_a) #para Haar de n_a qubits
            else:
                ancilla_rand_state = torch.ones((N_total, 1), dtype=torch.complex128) # Un escalar dummy para que el einsum no haga nada raro
        
        #ancillas para el conditioning
        cond_state = torch.zeros(2**self.n_c, dtype=torch.complex128)
        cond_state[0] = 1.0
        
        if prev_params is not None and prev_ensemble is not None:
            best_params = prev_params.clone()
            best_ensemble = prev_ensemble.clone()
            best_ensemble_probs = prev_probs.clone()
        else:
            best_params = torch.zeros([self.T, self.L, self.n_total, 2])
            best_ensemble = torch.zeros([N_total, 2**self.n,2**self.n, self.T+1], dtype=torch.complex128) 
            best_ensemble_probs = torch.zeros([N_total, 2**self.n, self.T+1], dtype=torch.complex128) 
            if Haar_initial is None:
                best_ensemble[:,:,:,T_start] = time_ensemble[:, :,:, T_start]
                best_ensemble_probs[:,:,T_start]=time_probs[:,:,T_start]
            else:
                Haar_ensemble=torch.zeros([N_total,2**self.n,2**self.n],dtype=torch.complex128)
                Haar_ensemble[:,:,0]=Haar_initial
                probs_Haar=torch.zeros([N_total,2**self.n],dtype=torch.float64)
                probs_Haar[:,0]=1.0
                best_ensemble[:,:,:,T_start] = Haar_ensemble
                best_ensemble_probs[:,:,T_start]=probs_Haar

        loss_hist = []
        best_loss_list = []
        
        if norm_in is None:
            norm=self._norm_loss(N,time_ensemble[0:N,:,:,T_start],time_probs[0:N,:,T_start],time_ensemble[:,:,:,0],time_probs[:,:,0]) #revisar
        else:
            norm=norm_in

        for t in range(T_start-1, -1, -1): #normal T-1, Haar T o 
            print(f"\n========== Denoising t={t} ==========")
            
            train_ensemble = best_ensemble[:, :,:, t+1].detach().clone()
            train_probs=best_ensemble_probs[:, :, t+1].detach().clone()
            target_ensemble = time_ensemble[:, :,:, t].detach().clone()
            target_probs = time_probs[:, :, t].detach().clone()
            
            thetas = torch.rand((self.L, self.n_total, 2), requires_grad=True, dtype=torch.float64)
            optimizer = torch.optim.Adam([thetas], lr=0.01)
            
            best_loss = float('inf')
            best_current_params = None 
            
            for step in range(0,steps+1):
                optimizer.zero_grad() #reinicia el gradiente del optimizador

                gen_probs,gen_ensemble = self._sample_states_from_circuit(N_total,thetas,train_ensemble,train_probs,mu,ancilla_rand_state,cond_state)
                
                loss = self._wasserstein_loss(N,target_ensemble,target_probs, gen_ensemble,gen_probs,norm)
                
                loss.backward() #obtiene los gradientes
                optimizer.step() 
                
                current_loss = loss.item()
                loss_hist.append(current_loss)
                
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_current_params = thetas.detach().clone()
                    
                if step % 100 == 0:
                    print(f"  Step {step} | Batch Loss: {current_loss:.6f}")
            best_loss_list.append(best_loss)
                    
            print("-> Best loss: ", best_loss)
            
            best_params[t, :,:,:] = best_current_params
            
            with torch.no_grad():
                best_gen_probs,best_gen_ensemble=self._sample_states_from_circuit(N_total,best_current_params,train_ensemble,train_probs,mu,ancilla_rand_state,cond_state)
                best_ensemble[:, :,:, t] = best_gen_ensemble
                best_ensemble_probs[:,:,t]=best_gen_probs

        return best_params, best_ensemble,best_ensemble_probs, loss_hist,best_loss_list,ancilla_rand_state,norm
        

    def backward_test(self,N,best_params,ensemble_in,probs_in,A): #states_haar
        salto=(2*math.pi)/self.Class
        mu_total=(torch.arange(self.Class)*salto).to(torch.float64)
        mu=mu_total[A]

        if self.n_a > 0:
            ancilla_state = generate_Haar_distribution(N, self.n_a) #para Haar de n_a qubits
        else:
            ancilla_state = torch.ones((N, 1), dtype=torch.complex128)

        cond_state = torch.zeros((2**self.n_c), dtype=torch.complex128)
        cond_state[0] = 1.0
        
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

import numpy as np
from scipy.linalg import expm, kron, eigvalsh
from scipy.optimize import minimize
import matplotlib.pyplot as plt

TEST=False


import torch
import torch.nn as nn
loss_err = nn.MSELoss()
@torch.no_grad()
def get_err_F_array(f_flat,A_flat, Ene_test, device='cpu'):
    '''From f and A, calculate energy, and compare to test data Ene_test
       f and A are arrays of data entries, to be computed in parallel
       All input are tensor on cuda, with dimension (num,*)
    '''

#def get_err_F_array(Ham_flat,F, Ene_test, device='cpu'):
    '''From Ham and F, calculate energy, and compare to test date Ene_test
       Ham and F are of array shape, to be computed in parallel
       All input are tensor on cuda, with dimension (num,*)
    '''
    # reshape into matrix form for matrix calculation
    num = f_flat.shape[0]

    Ham = Ham_flat.reshape((num,4,4))
    #print('Ham',Ham)
    #print('F',F)

    # Initialize matrices and vectors
    A = get_A()
    A = torch.tensor(A,device=device)
    v=get_v0()
    v = torch.tensor(v,device=device)

    F = F.type(torch.complex128)
    Ham = Ham.type(torch.complex128)

    # start calculation
    G=torch.einsum('ns,sjk->njk',F,A)
    #G = sum(F[i].cpu() * A[i] for i in range(6))
    #print('now got G',G)
    #print('v',v)
    #print('Ham',Ham)
    
    _eG = torch.linalg.matrix_exp(G)
    v = torch.einsum('naj,ij->nia',_eG,v)
    Ene = v2Ene(Ham,v)

    #print('Ene     ',Ene.real)
    #print('Ene_test',Ene_test)
    #print('Ene_diff',Ene_test-Ene.real)
    #print('Ene_diff',(Ene_test-Ene.real) <0.001)

    # compare all four terms
    # err = loss_err(Ene_test,Ene.real)
    # compare only the ground state energy
    #print((Ene_test.min(dim=1)).shape)
    #print((Ene.real.min(dim=1)).shape)
    e0=torch.min(Ene_test,dim=1).values
    e1=torch.min(Ene.real,dim=1).values
    print(e0)
    print(e1)
    err = loss_err(e0,e1)

    # compute average of Ene_pred
    Ene_pred_abs_mean = Ene.real.sum(dim=1).abs().mean().item()
    print(f'Ene_pred_abs_mean  = {Ene_pred_abs_mean}')
    Ene_pred_abs_mean2 = Ene.real.abs().mean().item()
    print(f'Ene_pred_abs_mean2 = {Ene_pred_abs_mean2}')
    
    return err.detach().cpu()


# Generate AllPauli array
def get_f_ham(PauliMatrix):
      if M==2:
         AllPauli = np.zeros((4, 4, 2**M, 2**M), dtype=complex)
         for i in range(4):
            for j in range(4):
               AllPauli[i, j] = kron(PauliMatrix[i], PauliMatrix[j])
         f = np.random.uniform(-10, 10, (4, 4))   #This is the input: a 4x4 matrix
         Aux = AllPauli * f[:, :, np.newaxis, np.newaxis]
         Ham = sum(Aux[i,j] for i in range (4) for j in range(4))
      elif M ==3:
         AllPauli = np.zeros((4, 4, 4, 2**M, 2**M), dtype=complex)
         for i in range(4):
            for j in range(4):
               for k in range(4):
                  AllPauli[i, j, k] = kron(kron(PauliMatrix[i], PauliMatrix[j]), PauliMatrix[k])
         f = np.random.uniform(-4, 4, (4, 4, 4))
         Aux = AllPauli * f[:, :, :, np.newaxis, np.newaxis]
         Ham = sum(Aux[i,j,k] for i in range (4) for j in range(4) for k in range(4))
      return f, Ham, AllPauli

#Number of qubits
M = 2
#Number of iterations
it = 2

def get_Pauli():
   # Define Pauli matrices
   I = np.eye(2, dtype=complex)
   PauliMatrix = [
      np.array([[1, 0], [0, 1]], dtype=complex),  # Pauli I
      np.array([[0, 1], [1, 0]], dtype=complex),  # Pauli X
      np.array([[0, -1j], [1j, 0]], dtype=complex),  # Pauli Y
      np.array([[1, 0], [0, -1]], dtype=complex)  # Pauli Z
   ]
   return PauliMatrix

def get_w():
   # Initialize w
   w = np.zeros((2**M))
   for i in range(2**M):
      w[i] = 2**M-i
   w = w/sum(w)
   return w

def generate(index=0):
   # Generate random real numbers in the range [-2, 2] and Hamiltonian
   import time
   _seed = int((time.time()*1e7 % 1e9))
   if TEST:
      np.random.seed(index) #for test
   else:
      np.random.seed(_seed+index*100)  #ensure random seeds for each parallel process

   PauliMatrix = get_Pauli()
   w = get_w()
   f, Ham, AllPauli = get_f_ham(PauliMatrix)
   
   # Initialize matrices and vectors containing the relevant data
   Ene = np.zeros((it+1, 2**M), dtype=complex)     # Energy
   v = np.zeros((2**M, 2**M), dtype=complex)       # The exponent
   A = np.zeros((it, 2**M*2**M), dtype=complex)    # Ansatz parameter

   for i in range(2**M):
      v[i,i] = 1

   # Define helper functions
   def expectation_value(ve1, AA):
      return np.vdot(ve1, AA @ ve1)

   def Be2(params1):
      Aux1 = AllPauli * params1[:, :, np.newaxis, np.newaxis]
      return sum(Aux1[i, j]*1j for i in range(4) for j in range(4))

   def Be3(params1):
      Aux1 = AllPauli * params1[:, :, :, np.newaxis, np.newaxis]
      return sum(Aux1[i, j, k]*1j for i in range(4) for j in range(4) for k in range(4))

   

   

   # Calculate initial expectation values
   for i in range(2**M):
      Ene[0, i] = expectation_value(v[i], Ham)

   # Perform the optimization loops
   for m in range(it):
      def fun_to_minimize(params1): 
         if M == 2: 
            return sum(w[j] * expectation_value(expm(Be2(np.reshape(params1,(4,4)))) @ v[j], Ham) for j in range(2**M)).real
         elif M== 3: 
            return sum(w[j] * expectation_value(expm(Be3(np.reshape(params1,(4,4,4)))) @ v[j], Ham) for j in range(2**M)).real   
      if M==2:
         res = minimize(fun_to_minimize, np.zeros(16))
         A[m] = res.x
         Aux2 = AllPauli * np.reshape(A[m],(4,4))[:, :, np.newaxis, np.newaxis]
         Antiunif = sum(Aux2[i, j]*1j for i in range (4) for j in range(4))
         for i in range(2**M):
            v[i] = expm(Antiunif) @ v[i]
            Ene[m + 1, i] = expectation_value(v[i], Ham)
      elif M==3:
         res = minimize(fun_to_minimize, np.zeros(64))
         A[m] = res.x
         Aux2 = AllPauli * np.reshape(A[m],(4,4,4))[:, :,:, np.newaxis, np.newaxis]
         Antiunif = sum(Aux2[i, j, k]*1j for i in range (4) for j in range(4) for k in range(4))
         for i in range(2**M):
            v[i] = expm(Antiunif) @ v[i]
            Ene[m + 1, i] = expectation_value(v[i], Ham).real


   # Printing the results
   if TEST:
      print("Exact    Energies:", eigvalsh(Ham))
      print("Starting Energies:", Ene[0].real)
      print("Approx   Energies:", Ene[1].real)
      print("Hamiltonian parameters (random) =", f) #This is the 4x4 input
      print("Ansatz paramaters =", np.reshape(A[0].real,(4,4))) #These are the output parameters, the 4x4 output
      #print("A[1] =", np.reshape(A[1].real,(4,4)))

   if __name__=="__main__":
      print('Start plotting')
      # Plot the results
      plt.plot(Ene[:, 0].real, label='Ene 1')
      plt.plot(Ene[:, 1].real, label='Ene 2')
      plt.plot(Ene[:, 2].real, label='Ene 3')
      plt.plot(Ene[:, 3].real, label='Ene 4')

      if M == 3:
         plt.plot(Ene[:, 4].real, label='Ene 5')
         plt.plot(Ene[:, 5].real, label='Ene 6')
         plt.plot(Ene[:, 6].real, label='Ene 7')
         plt.plot(Ene[:, 7].real, label='Ene 8')
         
      plt.legend()
      plt.xlabel('Iteration')
      plt.ylabel('Energy')
      plt.show()

   _data=[
      f,    # the initial seed for constructing Hamiltonian
      A.real,        # Ansatz parameter, only A.real[0] matters, others are zero
      A.imag,        # imag are zero
      Ene.real,   # the energy converges
      Ene.imag,   # the energy converges
      Ham.real, # the constructed hamiltonian  
      Ham.imag,             
      v.real,             # v contains the para for ground state, as the output
      v.imag,
    ]
   if True: 
      for _ in _data:
         print(_)
      print('print meta info for _data')      
      index=0
      for _ in _data:
         print(f"index:{index}, size:{_.size}, shape:{_.shape}")
         index += _.size
      '''
      index:0, size:16, shape:(4, 4)
      index:16, size:32, shape:(2, 16)
      index:48, size:32, shape:(2, 16)
      index:80, size:12, shape:(3, 4)
      index:92, size:12, shape:(3, 4)
      index:104, size:16, shape:(4, 4)
      index:120, size:16, shape:(4, 4)
      index:136, size:16, shape:(4, 4)
      index:152, size:16, shape:(4, 4)
      '''

   _data2 = [ _.flatten() for _ in _data ]
   data=np.concatenate( _data2 )
   #print(data)
   return data


if __name__=="__main__":
   TEST=True
   generate()

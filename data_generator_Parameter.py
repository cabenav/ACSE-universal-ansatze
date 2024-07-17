import numpy as np
from scipy.linalg import expm, kron, eigvalsh
from scipy.optimize import minimize
import matplotlib.pyplot as plt

TEST=False

def generate(index=0):
   # Generate random real numbers in the range [-2, 2] and Hamiltonian
   import time
   _seed = int((time.time()*1e7 % 1e9))
   if TEST:
      np.random.seed(index) #for test
   else:
      np.random.seed(_seed+index*100)  #ensure random seeds for each parallel process

   #Number of qubits
   M = 2
   #Number of iterations
   it = 2

   # Define Pauli matrices
   I = np.eye(2, dtype=complex)
   PauliMatrix = [
      np.array([[1, 0], [0, 1]], dtype=complex),  # Pauli I
      np.array([[0, 1], [1, 0]], dtype=complex),  # Pauli X
      np.array([[0, -1j], [1j, 0]], dtype=complex),  # Pauli Y
      np.array([[1, 0], [0, -1]], dtype=complex)  # Pauli Z
   ]

   # Initialize matrices and vectors containing the relevant data
   Ene = np.zeros((it+1, 2**M), dtype=complex)
   v = np.zeros((2**M, 2**M), dtype=complex)
   A = np.zeros((it, 2**M*2**M), dtype=complex)

   # Initialize w
   w = np.zeros((2**M))
   for i in range(2**M):
      w[i] = 2**M-i
   w = w/sum(w)

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

   # Generate AllPauli array
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
   if False:
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
        Ham.real, # the constructed hamiltonian  
        Ham.imag,       
        f.real,  # the initial seed for constructin Hamiltonian
        f.imag,
        Ene.real,   # the energy converges
        Ene.imag,   # the energy converges
        A.real,        # F goes to zero
        A.imag,        # F goes to zero
        v.real,             # v contains the para for ground state, as the output
        v.imag,
    ]
   if False: 
      print(_data)
      print('print meta info for _data')      
      index=0
      for _ in _data:
         print(f"index:{index}, size:{_.size}, shape:{_.shape}")
         index += _.size
      '''
      index:0, size:16, shape:(4, 4)
      index:16, size:16, shape:(4, 4)
      index:32, size:16, shape:(4, 4)
      index:48, size:16, shape:(4, 4)
      index:64, size:12, shape:(3, 4)
      index:76, size:12, shape:(3, 4)
      index:88, size:32, shape:(2, 16)
      index:120, size:32, shape:(2, 16)
      index:152, size:16, shape:(4, 4)
      index:168, size:16, shape:(4, 4)
      '''

   _data2 = [ _.flatten() for _ in _data ]
   data=np.concatenate( _data2 )
   #print(data)
   return data


if __name__=="__main__":
   generate()

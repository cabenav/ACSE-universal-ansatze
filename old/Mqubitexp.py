import numpy as np
from scipy.linalg import expm, kron, eigvalsh
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#Number of qubits
M = 2
#Number of iterations
it = 25

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
#F = np.zeros((it, 4), dtype=complex)
F = np.zeros((it, 2**M*2**M), dtype=complex)

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
   RR = np.random.uniform(-4, 4, (4, 4))
   Aux = AllPauli * RR[:, :, np.newaxis, np.newaxis]
   Ham = sum(Aux[i,j] for i in range (4) for j in range(4))
elif M ==3:
   AllPauli = np.zeros((4, 4, 4, 2**M, 2**M), dtype=complex)
   for i in range(4):
      for j in range(4):
         for k in range(4):
            AllPauli[i, j, k] = kron(kron(PauliMatrix[i], PauliMatrix[j]), PauliMatrix[k])
   RR = np.zeros((4, 4, 4))
   pp = np.random.uniform(-4,4,(4))
   RR[1,3,2] = pp[0]
   RR[2,2,3] = pp[1]
   RR[0,1,2] = pp[2]
   RR[1,1,2] = pp[3]
   Aux = AllPauli * RR[:, :, :, np.newaxis, np.newaxis]
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
          par = np.zeros((4,4,4),dtype=complex)
          par[1,3,2] = params1[0]
          par[2,2,3] = params1[1]
          par[0,1,2] = params1[2]
          par[1,1,2] = params1[3]
          return sum(w[j] * expectation_value(expm(Be3(par)*1j) @ v[j], Ham) for j in range(2**M)).real   
    if M==2:
       res = minimize(fun_to_minimize, np.zeros(16))
       F[m] = res.x
       Aux2 = AllPauli * np.reshape(F[m],(4,4))[:, :, np.newaxis, np.newaxis]
       Antiunif = sum(Aux2[i, j] for i in range (4) for j in range(4))
       for i in range(2**M):
          v[i] = expm(Antiunif) @ v[i]
          Ene[m + 1, i] = expectation_value(v[i], Ham)
    elif M==3:
       res = minimize(fun_to_minimize, np.ones(4))
       F[m] = res.x
       G = np.zeros((4,4,4),dtype=complex)
       G[1,3,2] = F[m,0]
       G[2,2,3] = F[m,1]
       G[0,1,2] = F[m,2]
       G[1,1,2] = F[m,3]
       Aux2 = AllPauli * G[:, :,:, np.newaxis, np.newaxis]
       Antiunif = sum(Aux2[i, j, k]*1j for i in range (4) for j in range(4) for k in range(4))
       for i in range(2**M):
          v[i] = expm(Antiunif) @ v[i]
          Ene[m + 1, i] = expectation_value(v[i], Ham)


# Printing the results

print("Exact Energies:", eigvalsh(Ham))
print("Approx Energies:", Ene[it-1])
print("Hamiltonian parameters:", RR)
print("'Ansatz' parameters:", F.real)


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




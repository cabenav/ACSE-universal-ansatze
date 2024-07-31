import numpy as np
from scipy.linalg import expm, kron, eigvalsh
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#Number of qubits
M = 2
#Number of iterations
it = 640

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
vv = np.zeros((2**M, 2**M), dtype=complex)
A = np.zeros((it+1, 2**M*2**M), dtype=complex)
B = np.zeros((it+1), dtype=complex)

# Initialize w
w = np.zeros((2**M))
for i in range(2**M):
   w[i] = 2**M-i
w = w/sum(w)
print(w)

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
AllPauli = np.zeros((4, 4, 2**M, 2**M), dtype=complex)
for i in range(4):
   for j in range(4):
      AllPauli[i, j] = kron(PauliMatrix[i], PauliMatrix[j])
f = np.random.uniform(1, 1, (4, 4)) #This is the input: a 4x4 matrix


# Perform the optimization loops
for yy in range (it+1):
   f[1,1] = -4 +yy/80
   Aux = AllPauli * f[:, :, np.newaxis, np.newaxis]
   Ham = sum(Aux[i,j] for i in range (4) for j in range(4))
   def fun_to_minimize(params1): 
      return sum(w[j] * expectation_value(expm(Be2(np.reshape(params1,(4,4)))) @ v[j], Ham) for j in range(2**M)).real 
   res = minimize(fun_to_minimize, np.zeros(16))
   A[yy] = res.x.real
   B[yy] = np.dot(A[yy],A[yy])
   Aux2 = AllPauli * np.reshape(A[yy],(4,4))[:, :, np.newaxis, np.newaxis]
   Antiunif = sum(Aux2[i, j]*1j for i in range (4) for j in range(4))
   for i in range(2**M):
      vv[i] = expm(Antiunif) @ v[i]
      Ene[yy, i] = expectation_value(vv[i], Ham)


# Printing the results

print("Exact    Energies:", eigvalsh(Ham))
print("Starting Energies:", Ene[0].real)
print("Approx   Energies:", Ene[1].real)
print("Hamiltonian parameters (random) =", f) #This is the 4x4 input
print("Ansatz paramaters =", np.reshape(A[0].real,(4,4))) #These are the output parameters, the 4x4 output
#print("A[1] =", np.reshape(A[1].real,(4,4)))

A = A.real
# Plot the results
plt.plot(Ene[:, 0].real, label='Ene 1')
plt.plot(Ene[:, 1].real, label='Ene 2')
plt.plot(Ene[:, 2].real, label='Ene 3')
plt.plot(Ene[:, 3].real, label='Ene 4')

plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.show()


plt.plot(A[:, 0].real, label='A 11')
plt.plot(A[:, 1].real, label='A 12')
plt.plot(A[:, 2].real, label='A 13')
plt.plot(A[:, 3].real, label='A 14')
plt.plot(A[:, 4].real, label='A 21')
plt.plot(A[:, 5].real, label='A 22')
plt.plot(A[:, 6].real, label='A 23')
plt.plot(A[:, 7].real, label='A 24')
plt.plot(A[:, 8].real, label='A 31')
plt.plot(A[:, 9].real, label='A 32')
plt.plot(A[:, 10].real, label='A 33')
plt.plot(A[:, 11].real, label='A 34')
plt.plot(A[:, 12].real, label='A 41')
plt.plot(A[:, 13].real, label='A 42')
plt.plot(A[:, 14].real, label='A 43')
plt.plot(A[:, 15].real, label='A 44')


  
plt.xlabel('Iteration')
plt.ylabel('Ansatz param')
plt.show()

plt.plot(B[:].real, label='HS')


  
plt.xlabel('Iteration')
plt.ylabel('Ansatz param')
plt.show()




import numpy as np
from scipy.linalg import expm, kron, eigvalsh
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define Pauli matrices
I = np.eye(2, dtype=complex)
PauliMatrix = [
    np.array([[1, 0], [0, 1]], dtype=complex),  # Pauli I
    np.array([[0, 1], [1, 0]], dtype=complex),  # Pauli X
    np.array([[0, -1j], [1j, 0]], dtype=complex),  # Pauli Y
    np.array([[1, 0], [0, -1]], dtype=complex)  # Pauli Z
]

# Define Ladder operators
Ladder = [
    np.array([[1, 0], [0, 0]], dtype=complex),  # Ladder[1]
    np.array([[0, 0], [0, 1]], dtype=complex)   # Ladder[2]
]

# Initialize A array
A = np.zeros((6, 4, 4), dtype=complex)

# Populate A array aka the basis of the Lie algebra
A[0] = kron(1j * PauliMatrix[2], Ladder[0])
A[1] = kron(1j * PauliMatrix[2], Ladder[1])
A[2] = kron(Ladder[0], 1j * PauliMatrix[2])
A[3] = kron(Ladder[1], 1j * PauliMatrix[2])
A[4][1, 2] = 1
A[4][2, 1] = -1
A[5][0, 3] = 1
A[5][3, 0] = -1

# Initialize matrices and vectors
Ene = np.zeros((4, 4), dtype=complex)
v = np.zeros((4, 4), dtype=complex)
F = np.zeros((3, 6), dtype=complex)
w = [0.4, 0.3, 0.2, 0.1]

v[0] = [1, 0, 0, 0]
v[1] = [0, 1, 0, 0]
v[2] = [0, 0, 1, 0]
v[3] = [0, 0, 0, 1]

# Define helper functions
def expectation_value(ve1, AA):
    return np.vdot(ve1, AA @ ve1)

def Be(a, b, c, d, e, f):
    return a * A[0] + b * A[1] + c * A[2] + d * A[3] + e * A[4] + f * A[5]

# Initialize w
w = np.array([0.4, 0.3, 0.2, 0.1])

# Generate AllPauli array
AllPauli = np.zeros((4, 4, 4, 4), dtype=complex)
for i in range(4):
    for j in range(4):
        AllPauli[i, j] = kron(PauliMatrix[i], PauliMatrix[j])

# Generate random real numbers in the range [-2, 2] and Hamiltonian
RR = np.random.uniform(-2, 2, (4, 4))
Aux = AllPauli * RR[:, :, np.newaxis, np.newaxis]
Ham = np.zeros((4, 4), dtype=complex)
for i in range(4):
    for j in range(4):
        Ham += Aux[i, j]

# Take the real part of Ham
Ham = Ham.real
# Print the matrix form of Ham


# Calculate initial expectation values
for i in range(4):
    Ene[0, i] = expectation_value(v[i], Ham)

# Perform the optimization loop
for m in range(3):
    def fun_to_minimize(params):
        a, b, c, d, e, f = params
        total = 0
        for j in range(4):
            total += w[j] *expectation_value(expm(Be(a, b, c, d, e, f)) @ v[j], Ham)
        return total.real    
    res = minimize(fun_to_minimize, np.zeros(6))
    F[m] = res.x
    G = sum(res.x[i] * A[i] for i in range(6))
    
    for i in range(4):
        v[i] = expm(G) @ v[i]
        Ene[m + 1, i] = expectation_value(v[i], Ham)

print(RR) 
print(F.real[0])
print(F.real[1])



print("Eigenvalues:", eigvalsh(Ham))
print("Computed eigenvalues:", Ene.real[1])
print("Hamiltonian parameters:", RR) 
print("'Ansatz' parameters:", F.real[0])
print("'Ansatz' parameters:", F.real[1])

# Plot the results
plt.plot(Ene[:, 0].real, label='Ene 1')
plt.plot(Ene[:, 1].real, label='Ene 2')
plt.plot(Ene[:, 2].real, label='Ene 3')
plt.plot(Ene[:, 3].real, label='Ene 4')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.show()

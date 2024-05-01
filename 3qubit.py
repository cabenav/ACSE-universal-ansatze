import numpy as np

# Define Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
identity = np.eye(2)

# Define Kronecker product of two matrices
def kronecker_product(a, b):
    return np.kron(a, b)

# Create the array of all possible Kronecker products of Pauli matrices
all_pauli = np.zeros((4, 4, 4), dtype=np.complex128)
for i in range(4):
    for j in range(4):
        for k in range(4):
            all_pauli[i, j, k] = kronecker_product(kronecker_product(sigma_x, sigma_x), sigma_x) if i == 0 \
                else kronecker_product(kronecker_product(sigma_x, sigma_x), sigma_y) if i == 1 \
                else kronecker_product(kronecker_product(sigma_x, sigma_x), sigma_z) if i == 2 \
                else kronecker_product(kronecker_product(sigma_x, sigma_x), identity)
            
            all_pauli[i, j, k] *= kronecker_product(sigma_y, sigma_y) if j == 0 \
                else kronecker_product(sigma_y, sigma_z) if j == 1 \
                else kronecker_product(sigma_y, identity)
            
            all_pauli[i, j, k] *= sigma_z if k == 0 \
                else identity

# Generate a random real tensor RR in the range [-10, 10]
RR = np.random.uniform(-10, 10, (4, 4, 4))

# Perform element-wise multiplication of all_pauli and RR
aux = all_pauli * RR

# Sum over all indices to get the Hamiltonian
Ham = np.zeros((8, 8), dtype=np.complex128)
for i in range(4):
    for j in range(4):
        for k in range(4):
            Ham += aux[i, j, k]

# Take the real part of the Hamiltonian
Ham = np.real(Ham)

print(Ham)
print(np.linalg.eigvals(Ham))

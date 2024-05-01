import numpy as np

# Define Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
identity = np.eye(2)

# Define the commutator function
def commutator(A, B):
    return np.dot(A, B) - np.dot(B, A)

# Define the expectation value function
def expectation_value(ve1, AA):
    return np.conjugate(ve1).dot(AA).dot(ve1)

# Create the array of all possible Kronecker products of Pauli matrices
all_pauli = np.array([[np.kron(sigma_x, sigma_x),
                        np.kron(sigma_x, sigma_y),
                        np.kron(sigma_x, sigma_z),
                        np.kron(sigma_x, identity)],
                       [np.kron(sigma_y, sigma_x),
                        np.kron(sigma_y, sigma_y),
                        np.kron(sigma_y, sigma_z),
                        np.kron(sigma_y, identity)],
                       [np.kron(sigma_z, sigma_x),
                        np.kron(sigma_z, sigma_y),
                        np.kron(sigma_z, sigma_z),
                        np.kron(sigma_z, identity)],
                       [np.kron(identity, sigma_x),
                        np.kron(identity, sigma_y),
                        np.kron(identity, sigma_z),
                        np.kron(identity, identity)]])

# Generate a random real matrix RR in the range [-2, 2]
RR = np.random.uniform(-2, 2, (4, 4))

# Perform element-wise multiplication of all_pauli and RR
aux = all_pauli * RR[:, :, np.newaxis, np.newaxis]

# Sum over the first two axes to get the Hamiltonian
ham = np.sum(aux, axis=(0, 1))

# Take the real part of the Hamiltonian (for simplicity but not needed in general)
ham = np.real(ham)

#print(ham)
#print(np.linalg.eigvals(ham))

# Define the ladder operator matrices
ladder = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]

# Initialize the array A with zeros
A = np.zeros((6, 4, 4), dtype=np.complex128)

# Lie algebra 
A[0] = np.kron(1j * sigma_y, ladder[0])
A[1] = np.kron(1j * sigma_y, ladder[1])
A[2] = np.kron(ladder[0], 1j * sigma_y)
A[3] = np.kron(ladder[1], 1j * sigma_y)
A[4][1, 2] = 1
A[4][2, 1] = -1
A[5][0, 3] = 1
A[5][3, 0] = -1

# Initialize vv and w arrays with zeros and specified values
vv = np.zeros((4, 4), dtype=np.complex128)
w = np.array([0.4, 0.3, 0.2, 0.1])

# Initialize arrays
Ene = np.zeros((100, 4))
Vec = np.zeros((101, 4, 4))
v = np.zeros((4, 4))

# Initialize v with standard basis vectors
v[0] = [1, 0, 0, 0]
v[1] = [0, 1, 0, 0]
v[2] = [0, 0, 1, 0]
v[3] = [0, 0, 0, 1]

eta = 0.29

# Main loop
for m in range(100):
    grad = np.zeros(6)
    vv = np.zeros((4, 4))
    for i in range(6):
        for j in range(4):
            grad[i] += np.real(w[j] * expectation_value(v[j], commutator(A[i], ham)))
    B = np.kron(0*ladder[0], 0*ladder[0])
    print(type(B))
    for i in range(6):
        B = B+ grad[i]*A[i]
    print(np.exp(eta * B))

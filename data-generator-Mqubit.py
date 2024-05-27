import numpy as np
from scipy.linalg import expm, kron, eigvalsh
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def generate(index=0):

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
    #w = [0.4, 0.3, 0.2, 0.1]

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
    import time
    _seed = int((time.time()*1e7 % 1e9))
    np.random.seed(_seed+index)
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
                total += w[j] * expectation_value(expm(Be(a, b, c, d, e, f)) @ v[j], Ham)
            return total.real    
        res = minimize(fun_to_minimize, np.zeros(6))
        F[m] = res.x
        G = sum(res.x[i] * A[i] for i in range(6))
    
        for i in range(4):
            v[i] = expm(G) @ v[i]
            Ene[m + 1, i] = expectation_value(v[i], Ham)
    '''
    print(RR) 
    print(F.real[0])
    print(F.real[1])

    print("Eigenvalues:", eigvalsh(Ham))
    print("Computed eigenvalues:", Ene.real[1])
    print("Hamiltonian parameters:", RR) 
    print("'Ansatz' parameters:", F.real[0])
    print("'Ansatz' parameters:", F.real[1])
    '''
    data=np.concatenate((
        RR.flatten(),
        Ham.flatten(),
        Ene.real[1],
        F.real[0],
    ))
    '''
    # data size
    print(
        RR.flatten().shape,
        Ham.flatten().shape,
        Ene.real[1].shape,
        F.real[0].shape,
    )
    print(data.shape)
    # output
    # (16,) (16,) (4,) (6,)
    # (42,)
    exit(0)
    '''
    #print(data)
    return data


############################# config start #################################
trials = 1000
# scipy use defult omp threads=4 for each process. hence the actual cpu being used is num_threads * 4
num_threads=16
# block to save data
block_size = num_threads * 200
folder='data'
filename_prefix=f'{folder}/m4'
#filename='tmp.npy'
# discontribute data into list of files with limited filesize or avoid slow I/O
filesize_limit = 50 # in Mb
filesize_limit_in_bytes = filesize_limit * 1.0e6 # in bytes
############################# config end  #################################

print(f'filename_prefix: {filename_prefix}-<index>.npy')
print(f'filesize_limit: {filesize_limit} Mb')
print(f'num_threads: {num_threads} (x 4 scipy threads) = {num_threads * 4}')
print(f'trials: {trials}')
print(f'block_size: {block_size}')

import os
def append(filename_prefix, array , filesize_limit = 100.0):
    data = array
    for i in range(1000):        
        filename = f'{filename_prefix}-{i}.npy'
        if os.path.exists(filename):
            if (os.path.getsize(filename) > filesize_limit_in_bytes):
                continue
            else:
                data_old = np.load(filename)
                data = np.concatenate((data_old, array))            
        np.save(filename, data)
        return filename, data.shape
    #print(f'{trial}/{trials} data {data.shape} appended into {filename} {data_new.shape}')

import sys
from multiprocessing import Pool
if __name__=="__main__":
    #generate(0)
    #exit()

    for trial in range(trials):
        with Pool(num_threads) as p:
            result = p.map(generate,list(range(block_size)))
            data = np.array(result)
            print('saving data...')
            filename, shape = append(filename_prefix,data)
            print(f'{trial}/{trials} data {data.shape} appended into {filename} {shape}')
                

    
    

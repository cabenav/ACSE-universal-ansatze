'''Construct Hamiltonian with random parameters, evaluate ansatz parameter and the ground state energy
   Parallel computing on CPU nodes, data saved in .npy format (readable by Numpy or Torch)
'''
# modified from Mqubit.py

import numpy as np
from scipy.linalg import expm, kron, eigvalsh
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import tqdm




def v2Ene(Ham,v):
    #print(torch.einsum('ij,j->i',AA , ve1)) # original Ham @ v, not in tensor/in parallel
    #Ene_pred =  torch.vdot(v, Ham @ v)
    Ham_v = torch.einsum('nij, nvj->nvi', Ham , v)  #Ham @ v
    Ene_pred =  torch.linalg.vecdot(v, Ham_v)
    return Ene_pred

def flat_v2Ene(Ham_flat,v_flat):    
    num = v_flat.shape[0]
    Ham = Ham_flat.reshape((num,4,4))
    v = v_flat.reshape((num,4,4))
    return v2Ene(Ham,v)
    
##  accuracy: reconstruct energy and calculate the absolute error
# this function is used to verified data. not used in data generation
def get_err(X_test,y_pred, Ene_test):
    #return torch.vdot(ve1, AA @ ve1)
    Ene_pred = flat_v2Ene(Ham_flat = X_test,v_flat=y_pred)
    #Ene_pred = v2Ene(v,Ham)
    
    #num = X_test.shape[0]
    #Ham = X_test.reshape((num,4,4))
    #v = y_pred.reshape((num,4,4))

    #print('v   ',v)    
    # an average error of 0.1 in v wourld lead err=2 in Ene
    #delta=1.02 #0.1 #1.79
    #print('delta:',delta)
    #v = v + delta
    #v = v * delta
    #print('v * ',v)
    

    print(Ene_test[:10])
    print(Ene_pred[:10])
    e0=Ene_test.sum(dim=1)
    e1=Ene_pred.sum(dim=1)
    err = ( (e1-e0)/e0 ).abs().mean()
    #ratio = (e1/e0).mean()
    #acc = 1 - (1 - ratio).abs()
    err = err.detach().cpu().item()
    #acc = acc.detach().cpu().item()
    #acc = acc * 100 # percentage diff
    print(e0[:10])
    print(e1[:10])
    print(e0.shape,e1.shape,err)
    #input()
    return err

# check if one can get the energy from v and Ham
def verify_data(d):
    print(d)
    d = d.clone().detach()
    d.to(device)
    #d = torch.tensor(d,device=device)
    Ham = d[:,:16]      #Ham as X
    v = d[:,76:92]    #v as y
    Ene=d[:,32:36]    #Energy
    err = get_err(Ham, v, Ene)
    #err = get_err(X_val, y_pred, Ene_test)
    print('err:',err)


def get_A():
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

    return A

def get_v0():
    v = np.zeros((4, 4), dtype=complex)
    v[0] = [1, 0, 0, 0]
    v[1] = [0, 1, 0, 0]
    v[2] = [0, 0, 1, 0]
    v[3] = [0, 0, 0, 1]
    return v

#def get_err_F(X_test,y_pred, Ene_test):
def get_err_F(Ham,F, Ene_test):
    # test for single entry
    Ham_flat = Ham[0]
    F=F[0]

    #num = Ham_flat.shape[0]
    #Ham = Ham_flat.reshape((num,4,4))
    Ham = Ham_flat.reshape((4,4))
    #v = v_flat.reshape((num,4,4))
    
    print('Ham',Ham)
    print('F',F)

    A = get_A()
    # Initialize matrices and vectors
    Ene = np.zeros((4, 4), dtype=complex)

    v=get_v0()

    # Define helper functions
    def expectation_value(ve1, AA):
        return np.vdot(ve1, AA @ ve1)

    def Be(a, b, c, d, e, f):
        return a * A[0] + b * A[1] + c * A[2] + d * A[3] + e * A[4] + f * A[5]


    # need v
    # given F, Ham, get the energy
    print(F[0],A[0])
    
    #_=torch.einsum('i,ijk->jk',F,A)
    #print(_)
    G = sum(F[i].cpu() * A[i] for i in range(6))
    print('now got G',G)
    #return
    m=0
    for i in range(4):
            v[i] = expm(G) @ v[i]
            print(v[i])
            print(Ham)
                  
            Ene[m + 1, i] = expectation_value(v[i], Ham.cpu())
    print(Ene)
    print(Ene_test)
    return

def get_err_F_array(Ham_flat,F, Ene_test):
    '''From Ham and F, calculate energy, and compare to test date Ene_test
       Ham and F are of array shape, to be computed in parallel
    '''
    # reshape into matrix form for matrix calculation
    num = Ham_flat.shape[0]
    Ham = Ham_flat.reshape((num,4,4))
    #print('Ham',Ham)
    #print('F',F)

    A = get_A()
    A = torch.tensor(A,device=device)

    # Initialize matrices and vectors
    Ene = np.zeros((num,4, 4), dtype=complex)

    v=get_v0()

    # Define helper functions
    def expectation_value(ve1, AA):
        return np.vdot(ve1, AA @ ve1)

    def Be(a, b, c, d, e, f):
        return a * A[0] + b * A[1] + c * A[2] + d * A[3] + e * A[4] + f * A[5]


    F = F.type(torch.complex128)
    #print(F)
    #print(A)
    G=torch.einsum('ns,sjk->njk',F,A)
    #G = sum(F[i].cpu() * A[i] for i in range(6))
    #print('now got G',G)

    v = torch.tensor(v,device=device)
    Ham = Ham.type(torch.complex128)
    #print('v',v)
    #print('Ham',Ham)
    
    _eG = torch.linalg.matrix_exp(G)
    v = torch.einsum('naj,ij->nia',_eG,v)
    Ene = v2Ene(Ham,v)

    print('Ene     ',Ene.real)
    print('Ene_test',Ene_test)
    return 


#generate one data entry as an 1D np array    
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
    if TEST:
        np.random.seed(index) #for test
    else:
        np.random.seed(_seed+index*100)  #ensure random seeds for each parallel process

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
        #print("*"*70)
        #print(res.x)
        #print(A)
    
        for i in range(4):
            v[i] = expm(G) @ v[i]
            Ene[m + 1, i] = expectation_value(v[i], Ham)


    
    #print("         Eigenvalues:", eigvalsh(Ham))
    #print("Computed eigenvalues:", Ene.real[1])
    #print("Hamiltonian parameters:", RR) 
    #print("'Ansatz' parameters:", F.real[0])
    #print("'Ansatz' parameters:", F.real[1])
    
    #print('v',v)
    _data=[
        Ham, # the constructed hamiltonian as input. probably need real and imaginary        
        RR,  # the initial seed
        Ene.real[2],   # the energy converges
        Ene.imag[2],   # the energy converges
        F.real,        # F goes to zero
        F.imag,        # F goes to zero
        v.real,             # v contains the para for ground state, as the output
        v.imag,
    ]
    #print(_data)
    _data2 = [ _.flatten() for _ in _data ]
    data=np.concatenate( _data2 )

    '''
    for mat of data
    flatten matrices with lengths [16, 16, 8, 36, 32]
    index info: {'total': 108, 'Ham': (0, 16), 'RR': (16, 32), 'Ene': (32, 40), 'F': (40, 76), 'v': (76, 108)}
    data.shape: (108,)
    model input: Ham
    model output: v
    '''    

    

    lengths=[16,16,4*2,6*3*2,16*2]
    keys=['Ham','RR','Ene','F','v']
    indexes = lengths.copy()
    indexes[0]=0
    for i in range(1,len(lengths)):
        indexes[i] = lengths[i-1] + indexes[i-1]    
    #print(data)
    
    info = {}
    info['total'] = indexes[-1]+lengths[-1]
    indexes.append(info['total'])
    for i in range(len(lengths)):
        info[keys[i]] = (indexes[i],indexes[i+1])
    print('lengths',lengths)
    #print(indexes)
    #print(keys)
    print('index info:',info)
    print('data.shape:',data.shape)
    print('input: Ham',Ham)
    print('output: v',v)
    print('F',F)

    

    #exit()
    
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
trials = 10000
# scipy use defult omp threads=4 for each process. hence the actual cpu being used is num_threads * 4
num_threads=16
# block to save data
block_size = num_threads * 200 * 5
folder='data'
#filename_prefix=f'{folder}/m4'
filename_prefix=f'{folder}/m6'
#filename='tmp.npy'
# discontribute data into list of files with limited filesize or avoid slow I/O
filesize_limit = 500 #50 # in Mb
filesize_limit_in_bytes = filesize_limit * 1.0e6 # in bytes
############################# config end  #################################
# data history
# m4: short version only contain RR and F
# m6: long version contain RR, Ham, v, F, Ene
# m7: same version with data verification

# result of verification
# (1) from v, one can construct Ene that matches eigen value of Ham
# (2) 0.01 error in v yields 100% error in Ene
# (3) RR give the same Ham, v and Ene everytime. This has been verified by fix random seeds

print(f'This code generate data and saves into filename_prefix: {filename_prefix}-<index>.npy')
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


TEST=True
    
import sys
from multiprocessing import Pool
if __name__=="__main__":
    if TEST:
        
        import torch
        device='cuda'
        row = generate(1)        
        #print('_'*80)
        #generate(1)
        #exit()
        
        _data = torch.tensor(row)
        data=torch.tensor(np.array([_data]))
        data=data.to(device)
        #verify_data(data)

        def error_test(data):
            d=data
            X = d[:,:16]
            y = d[:,76:92]    #parameters
            y = d[:,40:46]    #parameters #'F': (40, 76)
            #print(y)
            #exit()
            Ene_test=d[:,32:36]
            #def get_err(X_test,y_pred, Ene_test):
            print('_'*80)
            get_err_F(X,y, Ene_test)
            get_err_F_array(X,y, Ene_test)
        #exit()
        block_size=5
        with Pool(num_threads) as p:
            result = list(tqdm.tqdm(p.imap(generate, range(block_size)), total=block_size))
            #result = p.map(generate,list(range(block_size)))
            data = np.array(result)
        # program end without saving into file


        error_test(torch.tensor(data,device=device))
        exit()
        
        verify_data(data)

        exit()

    for trial in range(trials):
        with Pool(num_threads) as p:
            result = list(tqdm.tqdm(p.imap(generate, range(block_size)), total=block_size))            
            #result = p.map(generate,list(range(block_size)))
            data = np.array(result)
            filename, shape = append(filename_prefix,data)
            print(f'[{trial}/{trials}] data {data.shape} appended into {filename} {shape}')
                

    
    

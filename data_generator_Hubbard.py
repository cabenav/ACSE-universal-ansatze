import numpy as np
from numpy import linalg as LA
from numpy import count_nonzero
import math, cmath
from scipy.optimize import fmin, minimize, rosen, rosen_der
from itertools import product, combinations
from copy import copy
import matplotlib.pyplot as plt
from scipy import interpolate, linalg
from scipy.interpolate import make_lsq_spline, BSpline
from scipy.interpolate import make_interp_spline
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix
import numdifftools as nd
import scipy.optimize as optimize
import pickle
 
    
#NUMBER OF SITES, WEIGHTS and TROTTER STEPS:
#L = int(input("L number (integer) of sites: "))
#Num = int(input("Number of particles: "))
#trotter = int(input("Trotter (integer) steps: "))

L = 5 #5,8
Num = 2
trotter = 5
#u_input=0.3 # input value
#print(f"L={L},Num={Num},trotter={trotter},u_input={u_input}")

TEST=True
trials=500
num_threads = 2 #8  #currently 1 thread takes 1000% cpu. Total available cores are 128. hence 8 threads works fine
block_size=64 #512*8
filename_prefix = '/data/zwl/hubbard/L8n2-h10'


#FUNCTIONS
def Ham(H1,H2,U):
      return (H1+np.multiply(H2,U))

def vecf(w,res1):
   vecprov = []   
   for i in range(len(res1)):
      cont = 1     
      for x in range(len(w)):
         cont *= w[x]**(1-res1[i][x]) *(1-w[x])**(res1[i][x]) 
      vecprov.append(cont)
   return vecprov  

def chop(expr, delta=10**-6):
   return np.ma.masked_inside(expr, -delta, delta).filled(0).real

def sig(j):
   if j < L-1:   
      return j+1
   else:
      return 0
      
def sig2(j):
   if j > 3:   
      return j-1
   else:
      return j
      
def ant(j):
   if j == 0:   
      return 0
   elif j == 1:   
      return 0
   else:
      return j-2
      


#GENERATION OF THE HILBERT SPACE
## This generates the Hilbert space {|000>,|001>,...} 
vec = [ele for ele in product([1,0], repeat = L) if np.sum(ele)==Num]
vec = np.array(vec)
dimH = vec.shape[0]

#GENERATION OF THE OPERATORS
Op1 = np.zeros((L,dimH,dimH))
Op2 = np.zeros((L,dimH,dimH))

for j in range(L):
   for k1 in range(dimH):
      Op2[j,k1,k1] = vec[k1][j]
      if vec[k1][j] == 1 and vec[k1][sig(j)] == 0:
         aux = copy(vec[k1])
         aux[j] =0
         aux[sig(j)] = 1
         for k2 in range(dimH):
            if (aux == vec[k2]).all(): 
               if j < L-1:
                  Op1[j,k1,k2] = 1 
               else:
                  Op1[j,k1,k2] = (-1)**(Num-1) 

#CONSTRUCTION OF THE HAMILTONIANS

Ham1 =-sum(Op1[k] + np.transpose(Op1[k]) for k in range(L))
Ham2 = sum(np.matmul(Op2[k], Op2[sig(k)]) for k in range(L))
#Range = 20
#FI1 =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#Range = 20
#FI1 =[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]

Range = 10
FI1 =[0,1,2,3,4,5,6,7,8,9,10]
FI1b =[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

#CONSTRUCTION OF UNITARIES
def Unit(params,Hamil):
   x = params
   Full = sum(np.multiply(Op1[k]+np.transpose(Op1[k]),x[k])  for k in range(L))+ (1j)*sum(np.multiply(Op1[k]-np.transpose(Op1[k]),x[k+L])  for k in range(L))+sum(np.multiply(np.matmul(Op2[k], Op2[sig(k)]),x[k+2*L])  for k in range(L))
   Unitary = linalg.expm(-(1j)*Full)
   Unitarydag = np.conj(np.transpose(Unitary))
   return np.matmul(np.matmul(Unitarydag,Hamil),Unitary)
   
def Unit2(params):
   x = params
   Full = sum(np.multiply(Op1[k]+np.transpose(Op1[k]),x[k])  for k in range(L))+ (1j)*sum(np.multiply(Op1[k]-np.transpose(Op1[k]),x[k+L])  for k in range(L))+sum(np.multiply(np.matmul(Op2[k], Op2[sig(k)]),x[k+2*L])  for k in range(L))
   return linalg.expm(-(1j)*Full)
   
class function():
   def __init__(self,Hamil,vecL):
      self.Hamil = Hamil
      self.vecL = vecL
   def evalua(self,seed):
      matriz = Unit(seed,Hamil)
      return (np.matmul(np.matmul(np.conj(self.vecL),matriz),self.vecL)).real
   
def UnitH(params,Hamil):
   x = params
   Unitary = Unit2H(params)
   Unitarydag = np.conj(np.transpose(Unitary))
   return np.matmul(np.matmul(Unitarydag,Hamil),Unitary)
   
def Unit2H(params):
   x = params
   Full = sum(np.multiply(Op1[k]+np.transpose(Op1[k]),x[k])  for k in range(L))+sum(np.multiply(np.matmul(Op2[k], Op2[sig(k)]),x[k+L])  for k in range(L))
   return linalg.expm(Full)
   
   
class function2():
   def __init__(self,Hamil,vecL):
      self.Hamil = Hamil
      self.vecL = vecL
   def evalua(self,seed):
      matriz = UnitH(seed,self.Hamil)
      vec2 = Unit2H(seed) @ self.vecL
      return (np.matmul(np.matmul(np.conj(self.vecL),matriz),self.vecL)/(np.conj(vec2) @ vec2)).real



def run(u_input):
   eigen = [] 
   for u in range(0,Range+1):
      v1, v2 = LA.eig(Ham(Ham1,Ham2,u/2))
      eigen.append(v1)
      eigen[u].real
      eigen[u].sort()
   eigen=np.array(eigen)

   if False:
      eigen = [] 
      data_in_list=[]
      #for u in range(0,Range+1):
      min,max = 0,10
      for u in range(0,100000+1):
         _u = u/10000
         #h = Ham(Ham1,Ham2,_u/2)
         item_in = [1,1,1,1,1,_u/2,_u/2,_u/2,_u/2,_u/2]
         data_in_list.append(item_in)

         #print(_u/2,h)
         #input('...')

         v1, v2 = LA.eig(Ham(Ham1,Ham2,_u/2))
         r = v1.real
         r.sort()
         eigen.append(r)
         #eigen.append(v1)
         #eigen[u].real
         #eigen[u].sort()
      eigen=np.array(eigen)

      #print(eigen)
      print(eigen[:,0])
      data_out = eigen[:,0]
      data_out = data_out[...,np.newaxis]
      print(data_out)
      data_in = np.array(data_in_list)
      print(data_in)
      data=np.concatenate((data_in,data_out),1)
      print(data)
      np.save(filename:='eigen100000.npy',data)
      plt.xlabel('u/2')
      plt.ylabel('exact energy')
      plt.title('energy vs u \n Hamiltonian input: (1,1,1,1,1,u/2,u/2,u/2,u/2,u/2)')
      #plt.plot(range(0,10000+1)/10,data)
      x = np.linspace(start:=0,stop:=1.0001/2,num=10001)
      plt.plot(x,data_out)
      plt.show()
      #plt.plot(FI1b, (eigennumH[nn]-eigen[:,0])/eigen[:,0]*100, label=f"HCQE {nn}")


      #print(Ham1)
      #print(Ham2)
      input('...')

   #for i in range(dimH):
   #   plt.plot(FI1,np.transpose(eigen)[i],'r-', mfc='none',lw=1)
   #plt.rc('axes', labelsize=15)
   #plt.rc('font', size=15)  
   #plt.show()

   #xnew = np.linspace(0, 10, 300) 
   #for i in range(dimH):
   #   spl = make_interp_spline(FI1, np.transpose(eigen)[i], k=3)
   #   powersmooth = spl(xnew)
   #   plt.plot(xnew,powersmooth,'k-', mfc='none')
   #plt.rc('axes', labelsize=15)
   #plt.rc('font', size=15)  
   #plt.show()


   #QUANTUM ALGORITHM: here starts the quantum calculation

   #eigennum = np.zeros((trotter,Range+1))
   eigennumH = np.zeros((trotter,Range+1))
   decoherences_approx = np.zeros((trotter,Range+1))

   #seed=np.zeros((trotter,Range+1,3*L))
   seedH=np.zeros((trotter,Range+1,2*L))
   #frobenius = np.zeros((trotter,Range+1))
   frobeniusH = np.zeros((trotter,Range+1))

   state = np.zeros((Range+1,dimH),dtype=complex)
   for u in range(Range+1):
      state[u,0] =1
   state1 = np.zeros((Range+1,dimH),dtype=complex)
   for u in range(Range+1):
      state1[u,0] =1


   instate = np.zeros(2*L)
   for i in range(L):
      instate[i] = 1

   for nn in range(trotter):
      #for u in range(Range+1):
      for u in range(1): #only do it once
         #print("I am computing for the coupling: ", u, "  and the iteration: ", nn)
         _u = u_input/2
         #Hamil=Ham(Ham1,Ham2,u/2)
         Hamil=Ham(Ham1,Ham2,_u)

         res = minimize(function2(Hamil-np.identity(dimH)*eigen[u,0],state[u]).evalua,seedH[nn,u],method='L-BFGS-B')
         seedH[nn,u] = res.x   #output of the neural network : input of the unitary 
         #seedH[nn,u] = [1,1,1,1,1,u,u,u,u,u]
         frobeniusH[nn,u] = seedH[nn,u] @ seedH[nn,u]/L
         state[u] = Unit2H(seedH[nn,u]) @ state[u]       #computation of the new state
         state[u] = state[u]/np.sqrt((np.conj(state[u]) @ state[u]).real)      #normalization
         eigennumH[nn,u] = np.matmul(np.matmul(np.conj(state[u]),Hamil),state[u])             #energy calculation
         decoherences_approx[nn,u] = np.matmul(np.matmul(np.conj(state[u]),Ham1),state[u]) 
         if False: #nn==4:
            print('check')
            print(state[u])
            #print('u=',u,'state',state[u])
            print(Hamil)
         #res = minimize(function(Hamil,state[u]).evalua,seed[nn,u],method='L-BFGS-B')
         #seed[nn,u] = res.x
         #frobenius[nn,u] = seed[nn,u] @ seed[nn,u]
         #state[u] = Unit2(seed[nn,u]) @ state[u]
         #state[u] = state[u]/np.sqrt((np.conj(state[u]) @ state[u]).real)
         #eigennum[nn,u] = np.matmul(np.matmul(np.conj(state[u]),Hamil),state[u])
      plt.rc('axes', labelsize=15)
      plt.rc('font', size=15)
      plt.plot(FI1b, (eigennumH[nn]-eigen[:,0])/eigen[:,0]*100, label=f"HCQE {nn}")
      #plt.plot(FI1, (eigennum[nn]-eigen[:,0])/eigen[:,0]*100, label='ACQE')
      #plt.plot(FI1, eigennumH[nn], label='HCQE')
      #plt.plot(FI1, eigennum[nn], label='CQE')
   #plt.plot(FI1, eigen[:,0],'bo', mfc='none',label='exact')
   plt.legend(prop={"size":15},loc='upper left')
   plt.title("Error of the energy %")
   plt.xlabel("$U/t$")
   plt.show()


   #for u in range(Range+1):
   #u=0
   for i in range(L):
      instate[L+i] = u_input/2
   if False:
      u=0
      print("input Hamiltonian parameters", instate)
      for nn in range(trotter):
         print("output ansatz", nn, ":", seedH[nn,u])
      print("ground-state energy", eigennumH[nn,u])
      print("final state",state[0])
      #break
   #print('nn=',nn)
   #print(eigennumH[:,0])
   data1=[
      instate, #length-10 inputs for hamiltonian
      np.array([u_input]), #length-1 input for the program
      np.array([eigennumH[nn,0]]), #"ground-state energy"
      state[0].real, # the final state, length-10
      state[0].imag, 
      seedH[:,0].flatten(), #all ansatz parameters
   ]
   #print(data1)
   data=np.concatenate(data1,axis=0)
   #print(data)
   
   

   return data


   plt.rc('axes', labelsize=15)
   plt.rc('font', size=15)
   plt.title("Frobenius norm")
   plt.plot(FI1, frobeniusH[0], label='HCQE 0')
   plt.legend(prop={"size":15},loc='upper left')
   plt.xlabel("$U/t$")
   plt.show()

   plt.rc('axes', labelsize=15)
   plt.rc('font', size=15)  
   for nn in range(1,trotter):
      plt.plot(FI1, frobeniusH[nn], label=f"HCQE {nn}")
   plt.legend(prop={"size":15},loc='upper left')
   plt.xlabel("$U/t$")
   plt.show()

   #plt.rc('axes', labelsize=15)
   #plt.rc('font', size=15)  
   #for nn in range(trotter):
   #   plt.plot(FI1, frobenius[nn], label='CQE')
   #plt.legend(prop={"size":15},loc='upper left')
   #plt.xlabel("$U/t$")
   #plt.show()
   #pickle.dump(eigen, open( "list3.p", "wb" ) )
   #pickle.dump(eigennum, open( "list4.p", "wb" ) )


   #print inout, outpout and gs energy



#Ham1 =-sum(Op1[k] + np.transpose(Op1[k]) for k in range(L))
#Ham2 = sum(np.matmul(Op2[k], Op2[sig(k)]) for k in range(L))
def Xy2energy(_X,_y): #_X, _y for a single data entry
   #print(_X.shape,_y.shape)
   u_input= _X[-1]*2
   Hamil=Ham(Ham1,Ham2,u_input/2)
   #print(u_input)
   if _y.shape[0]==11:
      state = _y[1:]  # remove the first one for energy
   else:
      state = _y
   #print('check 2')
   #print(state)
   #print(Hamil)
   #print('conj',np.conj(state))
   eigennumH = np.matmul(np.matmul(np.conj(state),Hamil),state)
   
   if False: # compare
      energy = _y[0]  # not true when _y has length 10
      print('check diff:', eigennumH,energy,eigennumH-energy)
   return eigennumH
   



#from random import random
import random
from data_generator_Pool import append
from multiprocessing import Pool
import tqdm


from configurator import print_config
print_config(globals())


def energy_test():
   '''Do multiple test to compared computed energy with saved data, got an average error lower than 4e-16'''
   for i in range(1,100):
      e = 10/i
      d=run(e)
      X = d[:10]
      y = d[11:22]
      print(i,e,end=',')
      Xy2energy(X,y)
   

if __name__=="__main__":
   if TEST: # do a test with out save anything
      data=run(0.331)
      #print(data)
      d=data
      X = d[:10]
      y = d[11:22]
      Xy2energy(X,y)
      energy_test()
      
   else:
      # a parallel program to generate data and save into file incrementally
      for trial in range(trials):
         u_inputs = [ random.random() * 10 for _ in range(block_size)]
         
         with Pool(num_threads) as p:
               tqdm.tqdm()
               result = list(tqdm.tqdm(p.imap(run, u_inputs), total=block_size,desc=f"{trial}/{trials}"))            
               #result = p.map(generate,list(range(block_size)))
               data = np.array(result)
               filename, shape = append(filename_prefix,data)
               print(f'[{trial}/{trials}] data {data.shape} appended into {filename} {shape}')
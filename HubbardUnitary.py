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

def chop(expr, delta=10**-10):
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
   else:
      return j-1
      
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
    

#NUMBER OF SITES, WEIGHTS and TROTTER STEPS:
L = int(input("L number (integer) of sites: "))
Num = int(input("Number of particles: "))
trotter = int(input("Trotter (integer) steps: "))

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
Range = 20
FI1 =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

eigen = [] 
for u in range(0,Range+1):
   v1, v2 = LA.eig(Ham(Ham1,Ham2,u))
   eigen.append(v1)
   eigen[u].real
   eigen[u].sort()
eigen=np.array(eigen)

for i in range(dimH):
   plt.plot(FI1,np.transpose(eigen)[i],'r-', mfc='none',lw=1)
plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
plt.show()

#xnew = np.linspace(0, 10, 300) 
#for i in range(dimH):
#   spl = make_interp_spline(FI1, np.transpose(eigen)[i], k=3)
#   powersmooth = spl(xnew)
#   plt.plot(xnew,powersmooth,'k-', mfc='none')
#plt.rc('axes', labelsize=15)
#plt.rc('font', size=15)  
#plt.show()


#QUANTUM ALGORITHM: here starts the quantum calculation

eigennum = np.zeros((trotter,Range+1))
eigenor= np.zeros((trotter,Range+1))
eigennumor = np.zeros((trotter,Range+1))


seed=np.zeros((trotter,Range+1,3*L))
state = np.zeros((Range+1,dimH),dtype=complex)
for u in range(Range+1):
   state[u,0] =1

for nn in range(trotter):
   for u in range(Range+1):
      print("I am computing for the coupling: ", u, "  and the iteration: ", nn)
      Hamil=Ham(Ham1,Ham2,u)
      #seed[nn,u] = optimize.fmin(fun.evalua,seed[ant(nn),u],maxfun=800000,maxiter=800000,ftol=1e-6,xtol=1e-6)
      res = minimize(function(Hamil,state[u]).evalua,seed[nn,u],method='Powell')
      #Nelder-Mead, 
      seed[nn,u] = res.x
      state[u] = Unit2(seed[nn,u]) @ state[u]
      state[u] = state[u]/np.sqrt((np.conj(state[u]) @ state[u]).real)
      eigennum[nn,u] = np.matmul(np.matmul(np.conj(state[u]),Hamil),state[u])
   plt.rc('axes', labelsize=15)
   plt.rc('font', size=15)  
   plt.plot(FI1, eigennum[nn], label='CQE')
plt.plot(FI1, eigen[:,0],'bo', mfc='none',label='exact')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U/t$")
plt.show()





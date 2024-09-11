import numpy as np
from numpy import linalg as LA
from numpy import count_nonzero
import math, cmath
from scipy.optimize import fmin, minimize, rosen, rosen_der
from itertools import product, combinations
from copy import copy
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import make_lsq_spline, BSpline
from scipy.interpolate import make_interp_spline
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix
import numdifftools as nd
import scipy.optimize as optimize
import pickle
 
#FUNCTIONS

def dimensionH(Nn):
   cont = 0
   for i in range (Nn):
      cont += math.factorial(L)/(math.factorial(i)*math.factorial(L-i))
   return int(cont)


def expf(x,L):
   return complex(np.cos(2*np.pi*x/L),np.sin(2*np.pi*x/L))

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True 
    else:
        return False

def Ham(H1,H2,U):
   return (H1+np.multiply(H2,U))

def UnD(a1,a2,a3,a4):
   return np.matmul(np.matmul(np.matmul(np.transpose(Op[a1]),np.transpose(Op[a2])),Op[a3]),Op[a4])-np.matmul(np.matmul(np.matmul(np.transpose(Op[a4]),np.transpose(Op[a3])),Op[a2]),Op[a1])

def UnS(a1,a2):
   return np.matmul(np.transpose(Op[a1]),Op[a2])-np.matmul(np.transpose(Op[a2]),Op[a1])

def Unitary(th,OpAux):
   return np.identity(nf)+np.multiply(OpAux,np.sin(th))-np.multiply((np.cos(th)-1),np.matmul(OpAux,OpAux))


def vecf(w,res1):
   vec = []   
   for i in range(len(res1)):
      cont = 1     
      for x in range(len(w)):
         cont *= w[x]**res1[i][x]*(1-w[x])**(1-res1[i][x]) 
      vec.append(cont)
   return vec  
   
def sig(j):
   if j < L-1:   
      return j+1
   else:
      return 0
      
class function():
   def __init__(self, weig,res,Hamil):
      self.res = res
      self.Hamil = Hamil
      self.weig = weig
   def evalua(self,seed):
      matrizSD = Unit(seed,self.res,self.Hamil)
      elem = 0
      for ji in range(len(self.weig)):
         vec=np.zeros(len(self.weig))
         vec[ji]=1
         elem += self.weig[ji]*np.matmul(np.matmul(vec,matrizSD),vec)
      return elem
   def grad(self,seed):
      return nd.Gradient(self.evalua)(seed)
    

#NUMBER OF SITES, WEIGHTS and TROTTER STEPS:
L = int(input("L number (integer) of sites: "))
Num = int(input("Number of particles: "))
trotter = int(input("Trotter (integer) steps: "))
w = list(np.arange(0.5/L,0.5+0.01,0.5/L)) 
round_to_w = [round(num, 3) for num in w]

#GENERATION OF THE HILBERT SPACE
## This generates the Hilbert space {|000>,|001>,...} 
vec = [ele for ele in product([1,0], repeat = L) if np.sum(ele)==Num]
vec = np.array(vec)
dimH = vec.shape[0]

#GENERATION OF THE ANHILITATION OPERATORS
##Op[0],Op[1]... are the anhilitation operators for sites 0,1,...
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

eigen = [] 
entan = []
for u in range(11):
   v1, v2 = LA.eig(Ham(Ham1,Ham2,u))
   eigen.append(v1)
   eigen[u].real
   eigen[u].sort()

eigen=np.array(eigen)

FI1 =[0,1,2,3,4,5,6,7,8,9,10]
xnew = np.linspace(0, 10, 300) 

for i in range(dimH):
   plt.plot(FI1,np.transpose(eigen)[i],'r-', mfc='none',lw=1)
plt.show()


plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
for i in range(16):
   print(FI1,np.transpose(eigen)[i])
   spl = make_interp_spline(FI1, np.transpose(eigen)[i], k=3)
   powersmooth = spl(xnew)
   plt.plot(xnew,powersmooth,'k-', mfc='none')
   #plt.plot(FI1, eigen[:,i],'r-', mfc='none')
spl = make_interp_spline(FI1, eigen[:,nf-1], k=3)
powersmooth = spl(xnew)
plt.plot(xnew,powersmooth,'k-', mfc='none')
spl = make_interp_spline(FI1, eigen[:,0], k=3)
powersmooth = spl(xnew)
plt.plot(xnew,powersmooth,'r-', mfc='none',lw=4)
plt.show()

#CONSTRUCTION OF UNITARIES

#Generation of all Doubles Excitations
test_list = np.arange(0, L, 1).tolist()
res2 = list(combinations(test_list,2))
Doubles = []
for j1 in range(len(res2)):
   for k1 in range(j1+1,len(res2)):
      if(common_member(res2[j1],res2[k1])==False):
         #print(res2[j1],res2[k1],common_member(res2[j1],res2[k1]),j1,k1)
         Doubles.append((res2[j1],res2[k1]))

AllD = np.zeros((len(Doubles),nf,nf))
AllS = np.zeros((len(res),nf,nf))
#AllDsparse = []
#AllSsparse = []

OpS = []
for i in range(L+1):
   OpS.append(csr_matrix(Op[i]))


for j1 in range(len(Doubles)):
   AllD[j1] = UnD(Doubles[j1][0][0],Doubles[j1][0][1],Doubles[j1][1][0],Doubles[j1][1][1])[ni:ni+nf,ni:ni+nf]
   #AllDsparse.append(csr_matrix(AllD[j1]))

for j1 in range(len(res2)):
   AllS[j1] = UnS(res2[j1][0],res2[j1][1])[ni:ni+nf,ni:ni+nf]
   #AllSsparse.append(csr_matrix(AllS[j1]))


FI1 =[0,1,2,3,4,5, 6, 7, 8, 9,10]
FI1 = np.array(FI1)


#QUANTUM ALGORITHM: here starts the quantum calculation

#Generation of all Doubles Excitations
test_list = np.arange(0, L, 1).tolist()
res2 = list(combinations(test_list,2))
Doubles = []
for j1 in range(len(res2)):
   for k1 in range(j1+1,len(res2)):
      if(common_member(res2[j1],res2[k1])==False):
         #print(res2[j1],res2[k1],common_member(res2[j1],res2[k1]),j1,k1)
         Doubles.append((res2[j1],res2[k1]))

print(len(Doubles),len(res2))

def Unit(params,res2,Hamil):
   x = params
   Full = np.identity(nf)
   Full1 = np.identity(nf)
   FullS = np.identity(nf)
   Full1S = np.identity(nf)
   for j1 in range(len(Doubles)):
      Full = np.matmul(Unitary(x[j1],AllD[j1]),Full)
      Full1 = np.matmul(Full1, Unitary(-x[j1],AllD[j1]))
   for j1 in range(len(Doubles),2*len(Doubles)):
      j11 =  j1-len(Doubles)
      Full = np.matmul(Unitary(x[j1],AllD[j11]),Full)
      Full1 = np.matmul(Full1, Unitary(-x[j1],AllD[j11]))
   for j1 in range(2*len(Doubles),2*len(Doubles)+len(res2)):
      j11 = j1-2*len(Doubles)
      FullS = np.matmul(Unitary(x[j1],AllS[j11]),FullS)
      Full1S = np.matmul(Full1S, Unitary(-x[j1],AllS[j11]))
   for j1 in range(2*len(Doubles)+len(res2),2*len(Doubles)+2*len(res2)):
      j11 = j1-2*len(Doubles)-len(res2)
      FullS = np.matmul(Unitary(x[j1],AllS[j11]),FullS)
      Full1S = np.matmul(Full1S, Unitary(-x[j1],AllS[j11]))
   Full = np.matmul(LA.matrix_power(FullS,trotter),np.matmul(LA.matrix_power(Full, trotter),FullS))
   Full1 = np.matmul(np.matmul(Full1S,LA.matrix_power(Full1, trotter)),LA.matrix_power(Full1S,trotter))
   #Full = np.matmul(LA.matrix_power(np.matmul(FullS,Full),trotter),FullS)
   #Full1 = np.matmul(Full1S,LA.matrix_power(np.matmul(Full1,Full1S),trotter))
   return np.matmul(np.matmul(Full1,Hamil),Full)


def gradient_descent(gradient, start, learn_rate, n_iter, tolerance):
   vector = start
   for _ in range(n_iter):
      diff = -learn_rate * gradient(vector).real
      if np.all(np.abs(diff) <= tolerance):
         break
      vector += diff
   return vector


weights = vecf(w,res1)
eigennum = np.zeros((11,nf))
eigenor= np.zeros((11,nf))
eigennumor = np.zeros((11,nf))
gap = np.zeros(11)
gapnum = np.zeros(11)
gap2 = np.zeros(11)
gapnum2 = np.zeros(11)

seed=list(np.full(2*len(Doubles)+2*len(res2),0))
Hamil=Ham(Ham1,Ham2,0)

for u in range(11):
   print("I am computing the energies for the coupling u: ", u)
   Hamil=Ham(Ham1,Ham2,u)
   fun = function(weights[ni:ni+nf],res2,Hamil)
   #seed = gradient_descent(fun.grad,seed,0.2,20,1e-02)
   seed = optimize.fmin(fun.evalua, seed,maxfun=200000,maxiter=200000,ftol=1e-4,xtol=1e-4)
   vec=np.zeros(nf)
   vecaux=np.zeros(nf)
   for i in range(nf):
      vec=np.zeros(nf)
      vec[i]=1
      eigennum[u,i] = np.matmul(np.matmul(vec,Unit(seed,res2,Hamil)),vec)
   eigenor[u] = list(eigen.real[u])
   eigenor[u].sort() 
   eigennumor[u] = list(eigennum.real[u])
   eigennumor[u].sort()   
   gap[u] = eigenor[u,1]-eigenor[u,0]
   gapnum[u] = eigennumor[u][1]-eigennumor[u][0]
   gap2[u] = eigenor[u,2]-eigenor[u,0]
   gapnum2[u] = eigennumor[u][2]-eigennumor[u][0]
 

pickle.dump(eigen, open( "list3.p", "wb" ) )
pickle.dump(eigennum, open( "list4.p", "wb" ) )


plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
for i in range(nf-1):
   plt.plot(FI1, eigen[:,i],'bo', mfc='none')
   plt.plot(FI1, eigennum[:,i],'r*')
plt.plot(FI1, eigen[:,nf-1],'bo', mfc='none',label='exact')
plt.plot(FI1, eigennum[:,nf-1],'r*', label='UCC')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U/t$")
plt.show()

plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
plt.plot(FI1, gap,'bo', mfc='none',label='exact')
plt.plot(FI1, gapnum,'r*',label='UCC')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U/t$")
plt.show()

plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
plt.plot(FI1, gap2,'bo', mfc='none',label='exact')
plt.plot(FI1, gapnum2,'r*',label='UCC')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U/t$")
plt.show()


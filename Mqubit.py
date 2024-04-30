import numpy as np
from numpy import linalg as LA
from scipy.linalg import expm, sinm, cosm
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


#Unitary transformation of 1 qubit
def Unitary(x,y,z):
   return np.matrix([[np.cos(x/2), np.exp(z*1j)*np.sin(x/2)], [-np.exp((y-z)*1j)*np.sin(x/2), np.exp(y*1j)*np.cos(x/2)]])

#General Hamiltonian of 1 qubit
def Hamiltonian(a1,a2,a3,a4):
   return np.matrix([[a1+a2,a3+a4*1j], [a3-a4*1j, a1-a2]])

#General free Hamiltonian of 2 qubits = H \otimes 1 + 1 \otimes H 
def Free(a1,a2,a3,a4,b1,b2,b3,b4):
   return np.kron(Hamiltonian(a1,a2,a3,a4),np.identity(2)) +  np.kron(np.identity(2),Hamiltonian(b1,b2,b3,b4))

#General interacting Hamiltonian of 2 qubits = \sum_ij g_ij Pauli_i\otimes Pauli_j
def Interacting():
   #Pauli matrices
   s1 = np.matrix([[0,1],[1,0]]) 
   s2 = np.matrix([[0,-1j],[1j,0]])
   s3 = np.matrix([[1,0],[0,-1]])
   Pauli= [s1,s2,s3]
   Random= np.random.random((3, 3))*0.2  #g_ij
   A = np.asmatrix(np.zeros([4,4]))
   for i in range (3):
      for j in range(3):
         A = A + Random[i,j]*np.kron(Pauli[i],Pauli[j])
   return A

#Exponential unitary = Exp[\sum_i c_i Pauli_i\otimes Pauli_i]
def Aux(c0,c1,c2):
   s1 = np.matrix([[0,1],[1,0]])
   s2 = np.matrix([[0,-1j],[1j,0]])
   s3 = np.matrix([[1,0],[0,-1]])
   Pauli= [s1,s2,s3]
   Matr = c0*np.kron(Pauli[0],Pauli[0]) + c1*np.kron(Pauli[1],Pauli[1]) + c2*np.kron(Pauli[2],Pauli[2]) 
   return np.asmatrix(expm(-1j*Matr))

#Full Unitary of 2 qubits = U_1 \otimes U_2 Exp[\sum_i c_i Pauli_i\otimes Pauli_i] U_3 \otimes U_4
def FullUnitary(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3,z3,c0,c1,c2):
   Auxhere = np.matmul(np.kron(Unitary(x0,y0,z0),Unitary(x1,y1,z1)),Aux(c0,c1,c2))
   return np.matmul(Auxhere,np.kron(Unitary(x2,y2,z2),Unitary(x3,y3,z3)))

#Cost function of w-fields:
def cost(vec,FullH,w1,w2):
   v1 = np.matrix([1,0,0,0])
   v2 = np.matrix([0,1,0,0])
   v3 = np.matrix([0,0,1,0])
   v4 = np.matrix([0,0,0,1])
   Uni = FullUnitary(vec[0],vec[1],vec[2],vec[3],vec[4],vec[5],vec[6],vec[7],vec[8],vec[9], vec[10],vec[11],vec[12],vec[13],vec[14])
   cost1 = v1.dot(np.matmul(np.matmul(Uni.H,FullH),Uni)).dot(np.transpose(v1))
   cost2 = v2.dot(np.matmul(np.matmul(Uni.H,FullH),Uni)).dot(np.transpose(v2))
   cost3 = v3.dot(np.matmul(np.matmul(Uni.H,FullH),Uni)).dot(np.transpose(v3))
   cost4 = v4.dot(np.matmul(np.matmul(Uni.H,FullH),Uni)).dot(np.transpose(v4))
   LL = w1*w2*cost1+w1*(1-w2)*cost2+(1-w1)*w2*cost3+(1-w1)*(1-w2)*cost4
   return LL

def cost1(vec,FullH,w1,w2):
   v1 = np.matrix([1,0,0,0])
   v2 = np.matrix([0,1,0,0])
   v3 = np.matrix([0,0,1,0])
   v4 = np.matrix([0,0,0,1])
   Uni = FullUnitary(vec[0],vec[1],vec[2],vec[3],vec[4],vec[5],vec[6],vec[7],vec[8],vec[9], vec[10],vec[11],vec[12],vec[13],vec[14])
   cost1 = v1.dot(np.matmul(np.matmul(Uni.H,FullH),Uni)).dot(np.transpose(v1))
   return cost1

#Calculations of the Energies with a given set of parameters
def energies(vec,FullH):
   v1 = np.matrix([1,0,0,0])
   v2 = np.matrix([0,1,0,0])
   v3 = np.matrix([0,0,1,0])
   v4 = np.matrix([0,0,0,1])
   Uni = FullUnitary(vec[0],vec[1],vec[2],vec[3],vec[4],vec[5],vec[6],vec[7],vec[8],vec[9], vec[10],vec[11],vec[12],vec[13],vec[14])
   cost1 = v1.dot(np.matmul(np.matmul(Uni.H,FullH),Uni)).dot(np.transpose(v1))
   cost2 = v2.dot(np.matmul(np.matmul(Uni.H,FullH),Uni)).dot(np.transpose(v2))
   cost3 = v3.dot(np.matmul(np.matmul(Uni.H,FullH),Uni)).dot(np.transpose(v3))
   cost4 = v4.dot(np.matmul(np.matmul(Uni.H,FullH),Uni)).dot(np.transpose(v4))
   E1n = np.asarray(cost1)[0][0].real
   E2n = np.asarray(cost2)[0][0].real
   E3n = np.asarray(cost3)[0][0].real
   E4n = np.asarray(cost4)[0][0].real
   return [E1n,E2n,E3n,E4n]

class functioncost():
   def __init__(self, Hamilt,w1,w2):
      self.Hamilt = Hamilt
      self.w1 = w1
      self.w2 = w2
   def evalua(self,seed):
      return cost(seed,self.Hamilt,self.w1,self.w2).real

class functioncost1():
   def __init__(self, Hamilt,w1,w2):
      self.Hamilt = Hamilt
      self.w1 = w1
      self.w2 = w2
   def evalua(self,seed):
      return cost1(seed,self.Hamilt,self.w1,self.w2).real

#HAMILTONIAN (PARAMETERS AND CALCULATION)
p1 = 1
p2 = 2
p3 = 1
p4 = 3
q1 = 2
q2 = 1
q3 = 3
q4 = 5

i = 1
lista = []

while i < 501:
   Full = Free(p1,p2,p3,p4,q1,q2,q3,q4) + Interacting() #This computes the Hamiltonian
   w, v = LA.eig(Full)
   #print("Exact energies: ", w.real) #Eigenvalues
   #print("Exact states: ", v.real) #Eigenvectors
   #VARIATIONAL CALCULATION OF THE ENERGIES
   #1. Minimizating function cost (all the energies)
   seed=list(np.full(15,0))
   seed = optimize.fmin(functioncost(Full,0.7,0.9).evalua,seed,maxfun=200000,maxiter=200000,ftol=1e-12,xtol=1e-12)
   #print(seed)
   exact = np.sort(energies(seed,Full))
   #print("Calculated energies: ", exact)
   print(exact)
   print(np.sort(w.real))
   dif = [1-i/j for i,j in zip(exact,np.sort(w.real))]
   lista.append(list(dif))
   print("Differences: ",  dif)
   print("Average error: ", sum(dif)/len(dif))
   i+=1

pickle.dump(lista, open("list.p", "wb" ) )

#2. Minimizating function cost1 (only the ground state)
seed=list(np.full(15,0))
seed = optimize.fmin(functioncost1(Full,0.7,0.9).evalua, seed,maxfun=200000,maxiter=200000,ftol=1e-12,xtol=1e-12)
exact1 = np.sort(energies(seed,Full))
print("Calculated energies (ONLY GROUND-STATE CALCULATION!): ", exact1)
dif1 = [1-i/j for i,j in zip(exact1,np.sort(w.real))]
print("Differences: ",  dif1)
print("Average error: ", sum(dif1)/len(dif))

L = int(input("Ah, qué tal?"))



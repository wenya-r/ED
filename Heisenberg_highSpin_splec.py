#!/usr/bin/env python
# coding: utf-8

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d #Hilbert space spin basis
import numpy as np
import time 
from scipy.sparse.linalg import eigsh
#from mpi4py import MPI
import h5py
#from CeigenV import eigenc
from slepc_para.eigenV import eigenc
_time = time.time()
L = 16#number of sites
initE = -25
Jxy = np.sqrt(1.)
Jzz_0 = 1.
#hz = 0 #1//np.sqrt(3.)

k = L//2 
z = -1 
p = -1
print('k = ', k)
print('z = ', z)
basis = spin_basis_1d(L, pauli = False, S = '1', m = 0, kblock = k, pblock = p, zblock = z) 
# Define operators with Open Boundary Condition (OBC) using site-coupling lists
J_zz = [[Jzz_0, i, (i+1)%L] for i in range(L)]
J_xy = [[Jxy/2., i, (i+1)%L] for i in range(L)]
#h_z = [[hz, i] for i in range(L)]
#static and dynamic lists
static = [["+-", J_xy], ["-+", J_xy], ["zz", J_zz]]
dynamic = []
H = hamiltonian(static, dynamic, basis = basis, dtype = np.float64)
#print(H)
M = H.tocsr()
#print('shape=', M.shape)

E = eigenc(M.data, M.indices, M.indptr, basis.Ns)
#E,  V = H.eigsh(sigma = initE, k = 2 , which = "SA", maxiter = 1E9, ncv = 100, mode = 'cayley')
print('E = ', E)

print(time.time() - _time, "seconds")
print('num_elets', len(M.data))
A = h5py.File('highSpin_22sites_splec_1exc2_petsc.hdf5', 'w')
#eigenV = A.create_dataset('eigenvalues', ( 4, ), dtype = 'f')
#tim = A.create_dataset('time', (1,))
#Sites = A.create_dataset('L', (1, ), dtype = 'i')
#eigenVec1 = A.create_dataset('vectors0', ( 4 ,vec1), dtype = 'f8')
#k_value = A.create_dataset('k', (1,), dtype = 'i')
#z_value = A.create_dataset('z', (1,), dtype = 'i')
#size = A.create_dataset('Ns', (1,), dtype = 'i')

A['eigenvalues'] = E
A['time'] = time.time() - _time
A['num_ele'] = len(M.data)
A['k'] = k 
A['p'] = p
A['z'] = z
A['Ns'] = basis.Ns
A['L'] = L


print('Ns', basis.Ns)
A.close()



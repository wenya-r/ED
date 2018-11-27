from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d #Hilbert space spin basis
import numpy as np
from scipy.sparse.linalg import eigsh



def eigenValues(L, pair):
    Jxy = np.sqrt(1.)
    Jzz_0 = 1.
    hz = 0 #1//np.sqrt(3.)

# Define operators with Open Boundary Condition (OBC) using site-coupling lists
    J_zz = [[Jzz_0, i, (i+1)%L] for i in range(L)]
    J_xy = [[Jxy/2., i, (i+1)%L] for i in range(L)]
    h_z = [[hz, i] for i in range(L)]


#tatic and dynamic lists
    static = [["+-", J_xy], ["-+", J_xy], ["zz", J_zz]]
    dynamic = []
    m, block = pair
    basis = spin_basis_1d(L, pauli = False, S="1", m = m, pblock = block) # 

    # compute the time-dependent Heisenberg Hamiltonian
    H_XXZ = hamiltonian(static, dynamic, basis = basis, dtype = np.float64)

# calculate entire spectrum only
    if basis.Ns == 0:
        E = None
            
    elif basis.Ns < 3000:
        ma = H_XXZ.todense()
        E, V = eigsh(ma)
#calculate full eigenspace
    else:
        k = basis.Ns//30   #400
        ncv = basis.Ns//10
  # calculate minimum and maximum energy only
        E, V = H_XXZ.eigsh(sigma = -20.0, k = k, which = "SA", maxiter = 1E9, ncv = ncv, mode = 'cayley')
#            myFile.Write_ordered("k = %d, ncv = %d \n"%(k, ncv) )
#            print("eigshE", E)
    
    basisCount = basis.Ns*2 if m!=0 else basis.Ns
#        print("Total Sz:", m, "size:", basisCount)
#        myFile.Write_ordered("Total Sz: %f size: %d\n"%(m, basisCount))

    if basis.Ns:
        if m==0:
            E = E.tolist()
        else :
            E = E.tolist()*2
   

    return E, basisCount

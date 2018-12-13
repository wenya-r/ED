import sys, slepc4py
slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc
import numpy
import time

#@profile
def eigenc(data, indices, indptr, Ns, numE=1):
    opts = PETSc.Options()
    n = opts.getInt('n', Ns)

    _time2 = time.time()
    A = PETSc.Mat(); #A.create()
    A.createAIJ(size = [n, n], csr=(indptr, indices, data), comm=PETSc.COMM_WORLD )
# test



#
    #A.setSizes([n, n])
    A.setFromOptions()
    A.setUp()

#    print('time for loop =',time.time()-_time2, 'secs')
    A.assemble()

    E = SLEPc.EPS(); E.create()

    E.setOperators(A)
    E.setProblemType(SLEPc.EPS.ProblemType.HEP)
    E.setDimensions(nev=numE)


    history = []


    def monitor(eps, its, nconv, eig, err):
        history.append(err[nconv])
    #print(its, rnorm)
    E.setMonitor(monitor)

    E.setFromOptions()
    E.solve()

    Print = PETSc.Sys.Print

    Print()
    Print("******************************")
    Print("*** SLEPc Solution Results ***")
    Print("******************************")
    Print()

    its = E.getIterationNumber()
    Print( "Number of iterations of the method: %d" % its )

    eps_type = E.getType()
    Print( "Solution method: %s" % eps_type )

    nev, ncv, mpd = E.getDimensions()
    Print( "Number of requested eigenvalues: %d" % nev )

    tol, maxit = E.getTolerances()
    Print( "Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit) )

    nconv = E.getConverged()
    Print( "Number of converged eigenpairs %d" % nconv )
    Eigenvalues = []
    if nconv > 0:
	  # Create the results vectors
          vr, vi = A.createVecs()
	  #
          Print()
          Print("        k          ||Ax-kx||/||kx|| ")
          Print("----------------- ------------------")
          for i in range(nconv):
            k = E.getEigenpair(i, vr, vi)
            error = E.computeError(i)
            if k.imag != 0.0:
              Print( " %9f%+9f j %12g" % (k.real, k.imag, error) )
            else:
              Print( " %12f       %12g" % (k.real, error) )
            Eigenvalues.append(k.real)
    return Eigenvalues

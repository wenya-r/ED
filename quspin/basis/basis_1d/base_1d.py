from ..lattice import lattice_basis
from . import _check_1d_symm as _check
import numpy as _np
import scipy.sparse as _sp
from numpy import array,cos,sin,exp,pi
from numpy.linalg import norm,eigvalsh,svd
from scipy.sparse.linalg import eigsh
import warnings
from types import ModuleType

# this is how we encode which fortran function to call when calculating 
# the action of operator string

_dtypes={"f":_np.float32,"d":_np.float64,"F":_np.complex64,"D":_np.complex128}


_basis_op_errors={1:"opstr character not recognized.",
				-1:"attemping to use real hamiltonian with complex matrix elements.",
				-2:"index of operator not between 0 <= index <= L-1"}

class OpstrError(Exception):
	# this class defines an exception which can be raised whenever there is some sort of error which we can
	# see will obviously break the code. 
	def __init__(self,message):
		self.message=message
	def __str__(self):
		return self.message

class bitops:
	def __init__(self,ops_module,**blocks):
		def try_add(func_str,block):
			try:
				self.__dict__[func_str] = ops_module.__dict__[func_str]
			except KeyError:
				if blocks.get(block) is not None:
					raise AttributeError("module {} missing implementation of {}.".format(module.__name__,func))

		try_add("py_fliplr","pblock")
		try_add("py_shift","kblock")
		try_add("py_flip_all","zblock")
		try_add("py_flip_sublat_A","zAblock")
		try_add("py_flip_sublat_B","zBblock")


class basis_1d(lattice_basis):
	def __init__(self,basis_module,ops_module,L,Np=None,pars=None,count_particles=False,**blocks):
		if self.__class__.__name__ == "basis_1d":
			raise ValueError("This class is not intended"
							 " to be instantiated directly.")


		# getting arguments which are used in basis.
		kblock=blocks.get("kblock")
		zblock=blocks.get("zblock")
		zAblock=blocks.get("zAblock")
		zBblock=blocks.get("zBblock")
		pblock=blocks.get("pblock")
		pzblock=blocks.get("pzblock")
		a=blocks.get("a")

		if type(L) is not int or L is 0:
			raise TypeError('L must be a positive integer')

		if self.sps < 2:
			raise ValueError("invalid value for sps, sps >= 2.")


		if type(a) is not int:
			raise TypeError('a must be integer')

		# checking if a is compatible with L
		if(L%a != 0):
			raise ValueError('L must be interger multiple of lattice spacing a')

		if pblock is not None:
			if type(pblock) is not int: raise TypeError('pblock must be integer')
			if abs(pblock) != 1: raise ValueError("pblock must be +/- 1")

		if zblock is not None:
			if type(zblock) is not int: raise TypeError('zblock/sblock must be integer')
			if abs(zblock) != 1: raise ValueError("zblock/sblock must be +/- 1")

		if zAblock is not None:
			if type(zAblock) is not int: raise TypeError('zAblock must be integer')
			if abs(zAblock) != 1: raise ValueError("zAblock must be +/- 1")

		if zBblock is not None:
			if type(zBblock) is not int: raise TypeError('zBblock must be integer')
			if abs(zBblock) != 1: raise ValueError("zBblock must be +/- 1")

		if pzblock is not None:
			if type(pzblock) is not int: raise TypeError('pzblock/psblock must be integer')
			if abs(pzblock) != 1: raise ValueError("pzblock/psblock must be +/- 1")

		if kblock is not None and (a <= L):
			if type(kblock) is not int: raise TypeError('kblock must be integer')
			if a == L:
				warnings.warn("using momentum with L == a",stacklevel=5)
			kblock = kblock % (L//a)
			blocks["kblock"] = kblock
			self._k = 2*pi*a*kblock/L

		self._L = L
		Ns = basis_module.get_Ns(L,Np,self.sps,**blocks) # estimate how many states in H-space to preallocate memory.
		self._basis_type = basis_module.get_basis_type(L,Np,self.sps,**blocks) # get the size of the integer representation needed for this basis (uint32,uint64,object)
		self._pars = _np.asarray(pars,dtype=self._basis_type)
		self._bitops = bitops(basis_module,**blocks)
		self._check_pcon = False
		if Np is None:
			self._conserved = ""
			self._Ns_pcon = None
			self._get_proj_pcon = False

		else:
			if type(Np) is not list:
				raise ValueError("basis_1d expects list for Np")

			self._Ns_pcon = None
			if len(Np)==1:
				self._Ns_pcon = basis_module.get_Ns(L,Np,self.sps,**{})
				self._check_pcon = True

			self._Nps = Np
			self._conserved = "N"
			self._make_n_basis = basis_module.n_basis
			self._get_proj_pcon = True

		# shout out if pblock and zA/zB blocks defined simultaneously
		if type(pblock) is int and ((type(zAblock) is int) or (type(zBblock) is int)):
			raise ValueError("zA and zB symmetries incompatible with parity symmetry")

		self._blocks_1d = blocks
		self._unique_me = True

		if count_particles: 
			Np_list = _np.zeros((Ns,),dtype=_np.int8)
		else:
			Np_list = None

		N,M = None,None

		if (type(kblock) is int) and (type(pblock) is int) and (type(zblock) is int):
			if self._conserved: self._conserved += " & T & P & Z"
			else: self._conserved = "T & P & Z"

			self._blocks_1d["pzblock"] = pblock*zblock
			self._unique_me = False

			basis=_np.empty((Ns,),dtype=self._basis_type)
			self._op = ops_module.t_p_z_op

			if self._basis_type == _np.object:
				# if object is basis type then most likely this is for single particle stuff in which case the 
				# normalizations need to be large ~ 1000 or more which won't fit in int8/int16.
				N=_np.empty(basis.shape,dtype=_np.int32) 
				M=_np.empty(basis.shape,dtype=_np.uint32)
			else:
				N=_np.empty(basis.shape,dtype=_np.int8) # normalisation*sigma
				M=_np.empty(basis.shape,dtype=_np.uint16) # m = mp + (L+1)mz + (L+1)^2c; Anders' paper

			if Np is None:
				Ns = basis_module.t_p_z_basis(L,pblock,zblock,kblock,a,self._pars,N,M,basis)
			else:
				# arguments get overwritten by ops.-_basis 
				Ns = basis_module.n_t_p_z_basis(L,Np,pblock,zblock,kblock,a,self._pars,N,M,basis,Np_list)

			self._Ns = Ns

		elif (type(kblock) is int) and (type(zAblock) is int) and (type(zBblock) is int):
			if self._conserved: self._conserved += " & T & ZA & ZB"
			else: self._conserved = "T & ZA & ZB"
			self._blocks_1d["zblock"] = zAblock*zBblock


			basis=_np.empty((Ns,),dtype=self._basis_type)
			self._op = ops_module.t_zA_zB_op

			if self._basis_type == _np.object:
				N=_np.empty(basis.shape,dtype=_np.int32) 
				M=_np.empty(basis.shape,dtype=_np.uint32)
			else:
				N=_np.empty(basis.shape,dtype=_np.int8)
				M=_np.empty(basis.shape,dtype=_np.uint16)

			if Np is None:
				Ns = basis_module.t_zA_zB_basis(L,zAblock,zBblock,kblock,a,self._pars,N,M,basis)
			else:
				Ns = basis_module.n_t_zA_zB_basis(L,Np,zAblock,zBblock,kblock,a,self._pars,N,M,basis,Np_list)				
				
			self._Ns = Ns

		elif (type(kblock) is int) and (type(pzblock) is int):
			if self._conserved: self._conserved += " & T & PZ"
			else: self._conserved = "T & PZ"
			self._unique_me = False

			basis=_np.empty((Ns,),dtype=self._basis_type)
			self._op = ops_module.t_pz_op

			if self._basis_type == _np.object:
				N=_np.empty(basis.shape,dtype=_np.int32) 
				M=_np.empty(basis.shape,dtype=_np.uint32)
			else:			
				N=_np.empty(basis.shape,dtype=_np.int8)
				M=_np.empty(basis.shape,dtype=_np.uint8) #mpz

			if Np is None:
				Ns = basis_module.t_pz_basis(L,pzblock,kblock,a,self._pars,N,M,basis)
			else:
				Ns = basis_module.n_t_pz_basis(L,Np,pzblock,kblock,a,self._pars,N,M,basis,Np_list)
				
			self._Ns = Ns

		elif (type(kblock) is int) and (type(pblock) is int):
			if self._conserved: self._conserved += " & T & P"
			else: self._conserved = "T & P"
			self._unique_me = False

			basis=_np.empty((Ns,),dtype=self._basis_type)
			self._op = ops_module.t_p_op

			if self._basis_type == _np.object:
				N=_np.empty(basis.shape,dtype=_np.int32) 
				M=_np.empty(basis.shape,dtype=_np.uint32)
			else:			
				N=_np.empty(basis.shape,dtype=_np.int8)
				M=_np.empty(basis.shape,dtype=_np.uint8)

			if Np is None:
				Ns = basis_module.t_p_basis(L,pblock,kblock,a,self._pars,N,M,basis)
			else:
				Ns = basis_module.n_t_p_basis(L,Np,pblock,kblock,a,self._pars,N,M,basis,Np_list)

			self._Ns = Ns

		elif (type(kblock) is int) and (type(zblock) is int):
			if self._conserved: self._conserved += " & T & Z"
			else: self._conserved = "T & Z"
			basis=_np.empty((Ns,),dtype=self._basis_type)
			self._op = ops_module.t_z_op

			if self._basis_type == _np.object:
				N=_np.empty(basis.shape,dtype=_np.int32) 
				M=_np.empty(basis.shape,dtype=_np.uint32)
			else:
				N=_np.empty(basis.shape,dtype=_np.int8)
				M=_np.empty(basis.shape,dtype=_np.uint8)

			if Np is None:
				Ns = basis_module.t_z_basis(L,zblock,kblock,a,self._pars,N,M,basis)
			else:
				Ns = basis_module.n_t_z_basis(L,Np,zblock,kblock,a,self._pars,N,M,basis,Np_list)				
				
			self._Ns = Ns

		elif (type(kblock) is int) and (type(zAblock) is int):
			if self._conserved: self._conserved += " & T & ZA"
			else: self._conserved = "T & ZA"
			basis=_np.empty((Ns,),dtype=self._basis_type)
			self._op = ops_module.t_zA_op

			if self._basis_type == _np.object:
				N=_np.empty(basis.shape,dtype=_np.int32) 
				M=_np.empty(basis.shape,dtype=_np.uint32)
			else:			
				N=_np.empty(basis.shape,dtype=_np.int8)
				M=_np.empty(basis.shape,dtype=_np.uint8)

			if Np is None:
				Ns = basis_module.t_zA_basis(L,zAblock,kblock,a,self._pars,N,M,basis)
			else:
				Ns = basis_module.n_t_zA_basis(L,Np,zAblock,kblock,a,self._pars,N,M,basis,Np_list)				
				
			self._Ns = Ns

		elif (type(kblock) is int) and (type(zBblock) is int):
			if self._conserved: self._conserved += " & T & ZB"
			else: self._conserved = "T & ZB"
			basis=_np.empty((Ns,),dtype=self._basis_type)
			self._op = ops_module.t_zB_op

			if self._basis_type == _np.object:
				N=_np.empty(basis.shape,dtype=_np.int32) 
				M=_np.empty(basis.shape,dtype=_np.uint32)
			else:			
				N=_np.empty(basis.shape,dtype=_np.int8)
				M=_np.empty(basis.shape,dtype=_np.uint8)

			if Np is None:
				Ns = basis_module.t_zB_basis(L,zBblock,kblock,a,self._pars,N,M,basis)
			else:
				Ns = basis_module.n_t_zB_basis(L,Np,zBblock,kblock,a,self._pars,N,M,basis,Np_list)
			
			self._Ns = Ns

		elif (type(pblock) is int) and (type(zblock) is int):
			if self._conserved: self._conserved += " & P & Z"
			else: self._conserved += "P & Z"
			basis = _np.empty((Ns,),dtype=self._basis_type)
			#print("line 300, basis =", basis)
			N=_np.empty((Ns,),dtype=_np.int8)
			self._op = ops_module.p_z_op
			
			if Np is None:
				Ns = basis_module.p_z_basis(L,pblock,zblock,self._pars,N,basis)
			else:
				Ns = basis_module.n_p_z_basis(L,Np,pblock,zblock,self._pars,N,basis,Np_list)

			self._Ns = Ns

		elif (type(zAblock) is int) and (type(zBblock) is int):
			if self._conserved: self._conserved += " & ZA & ZB"
			else: self._conserved += "ZA & ZB"

			self._op = ops_module.zA_zB_op

			basis = _np.empty((Ns,),dtype=self._basis_type)
			N=_np.empty((Ns,),dtype=_np.int8)
			if Np is None:
				Ns = basis_module.zA_zB_basis(L,zAblock,zBblock,self._pars,N,basis)
			else:
				Ns = basis_module.n_zA_zB_basis(L,Np,zAblock,zBblock,self._pars,N,basis,Np_list)

			self._Ns = Ns

		elif type(pblock) is int:
			if self._conserved: self._conserved += " & P"
			else: self._conserved = "P"
			basis = _np.empty((Ns,),dtype=self._basis_type)
			#print( "line 330, basis = ", basis)
			N=_np.empty((Ns,),dtype=_np.int8)
			self._op = ops_module.p_op


			#print("N = ", N)
			#print("Np =", Np)
			if Np is None:
				Ns = basis_module.p_basis(L,pblock,self._pars,N,basis)
			else:
				Ns = basis_module.n_p_basis(L,Np,pblock,self._pars,N,basis,Np_list)

			self._Ns = Ns
			#print("line 330, basis = ", basis)

			#print("N = ", N)
			#print("Np =", Np)
		elif type(zblock) is int:
			if self._conserved: self._conserved += " & Z"
			else: self._conserved += "Z"
			basis = _np.empty((Ns,),dtype=self._basis_type)
			N=_np.empty((Ns,),dtype=_np.int8)
			self._op = ops_module.z_op

			if Np is None:
				Ns = basis_module.z_basis(L,zblock,self._pars,N,basis)
			else:
				Ns = basis_module.n_z_basis(L,Np,zblock,self._pars,N,basis,Np_list)

			self._Ns = Ns

		elif type(zAblock) is int:
			if self._conserved: self._conserved += " & ZA"
			else: self._conserved += "ZA"
			basis = _np.empty((Ns,),dtype=self._basis_type)
			N=_np.empty((Ns,),dtype=_np.int8)
			self._op = ops_module.zA_op

			if Np is None:
				Ns = basis_module.zA_basis(L,zAblock,self._pars,N,basis)
			else:
				Ns = basis_module.n_zA_basis(L,Np,zAblock,self._pars,N,basis,Np_list)

			self._Ns = Ns

		elif type(zBblock) is int:
			if self._conserved: self._conserved += " & ZB"
			else: self._conserved += "ZB"
			basis = _np.empty((Ns,),dtype=self._basis_type)
			N=_np.empty((Ns,),dtype=_np.int8)
			self._op = ops_module.zB_op

			if Np is None:
				Ns = basis_module.zB_basis(L,zBblock,self._pars,N,basis)
			else:
				Ns = basis_module.n_zB_basis(L,Np,zBblock,self._pars,N,basis,Np_list)

			self._Ns = Ns
			
		elif type(pzblock) is int:
			if self._conserved: self._conserved += " & PZ"
			else: self._conserved += "PZ"
			basis = _np.empty((Ns,),dtype=self._basis_type)
			N=_np.empty((Ns,),dtype=_np.int8)
			self._op = ops_module.pz_op

			if Np is None:
				Ns = basis_module.pz_basis(L,pzblock,self._pars,N,basis)
			else:
				Ns = basis_module.n_pz_basis(L,Np,pzblock,self._pars,N,basis,Np_list)

			self._Ns = Ns
	
		elif type(kblock) is int:
			if self._conserved: self._conserved += " & T"
			else: self._conserved = "T"
			basis=_np.empty((Ns,),dtype=self._basis_type)
			self._op = ops_module.t_op
			
			if self._basis_type == _np.object:
				N=_np.empty(basis.shape,dtype=_np.int32) 
			else:			
				N=_np.empty(basis.shape,dtype=_np.int8)

			if Np is None:
				Ns = basis_module.t_basis(L,kblock,a,self._pars,N,basis)
			else:
				Ns = basis_module.n_t_basis(L,Np,kblock,a,self._pars,N,basis,Np_list)

			self._Ns = Ns

		else: 
			if Np is None:
				self._op = ops_module.op
				self._basis = _np.arange(Ns-1,-1,-1,dtype=self._basis_type)
				self._Ns = Ns
			else:
				self._op = ops_module.n_op
				basis = _np.empty((Ns,),dtype=self._basis_type)
				basis_module.n_basis(L,Np,self._pars,basis,Np_list)
				self._Ns = Ns

		if N is not None and M is not None:
			if Np is None or len(Np)==1:
				if Ns > 0:
					self._N = N[Ns-1::-1].copy()
					self._M = M[Ns-1::-1].copy()
					self._basis = basis[Ns-1::-1].copy()
					if Np_list is not None: self._Np_list = Np_list[Ns-1::-1].copy()
				else:
					self._N = _np.array([],dtype=N.dtype)
					self._M = _np.array([],dtype=M.dtype)
					self._basis = _np.array([],dtype=basis.dtype)
					if Np_list is not None: self._Np_list = _np.array([],dtype=Np.dtype)
			else:
				if self._unique_me:
					arg = _np.argsort(basis[:Ns],kind="heapsort")[::-1]
				else:
					arg = _np.argsort(basis[:Ns],kind="mergesort")[::-1]

				self._basis = basis[arg].copy()
				self._N = N[arg].copy()
				self._M = M[arg].copy()
				if Np_list is not None: self._Np_list = Np_list[arg].copy()

			self._op_args=[self._N,self._M,self._basis,self._L,self._pars]

		elif N is not None:
			if Np is None or len(Np)==1:
				if Ns > 0:
					self._N = N[Ns-1::-1].copy()
					self._basis = basis[Ns-1::-1].copy()
					if Np_list is not None: self._Np_list = Np_list[Ns-1::-1].copy()
				else:
					self._N = _np.array([],dtype=N.dtype)
					self._basis = _np.array([],dtype=basis.dtype)
					if Np_list is not None: self._Np_list = _np.array([],dtype=Np.dtype)
			else:
				arg = _np.argsort(basis[:Ns],kind="heapsort")[::-1]

				self._basis = basis[arg].copy()
				self._N = N[arg].copy()
				if Np_list is not None: self._Np_list = Np_list[arg].copy()

			self._op_args=[self._N,self._basis,self._L,self._pars]

		else:
			if Np is None: 
				pass
			elif len(Np)==1:
				self._basis = basis[Ns-1::-1].copy()
				if Np_list is not None: self._Np_list = Np_list[Ns-1::-1].copy()
			else:
				arg = _np.argsort(basis[:Ns],kind="heapsort")[::-1]

				self._basis = basis[arg].copy()
				if Np_list is not None: self._Np_list = Np_list[arg].copy()

			self._op_args=[self._basis,self._pars]




	@property
	def L(self):
		"""int: length of lattice."""
		return self._L

	@property
	def N(self):
		"""int: number of sites the basis is constructed with."""
		return self._L



	@property
	def description(self):
		"""str: information about `basis` object."""
		blocks = ""
		lat_space = "lattice spacing: a = {a}".format(**self._blocks)

		for symm in self._blocks:
			if symm != "a":
				blocks += symm+" = {"+symm+"}, "

		blocks = blocks.format(**self._blocks)

		if len(self._conserved) == 0:
			symm = "no symmetry"
		elif len(self._conserved) == 1:
			symm = "symmetry"
		else:
			symm = "symmetries"

		string = """1d basis for chain of length L = {0} containing {5} states \n\t{1}: {2} \n\tquantum numbers: {4} \n\t{3} \n\n""".format(self._L,symm,self._conserved,lat_space,blocks,self._Ns)
		string += self.operators
		return string 

	def _Op(self,opstr,indx,J,dtype):

		indx = _np.asarray(indx,dtype=_np.int32)
		
		if len(opstr) != len(indx):
			raise ValueError('length of opstr does not match length of indx')

		if _np.any(indx >= self.N) or _np.any(indx < 0):
			raise ValueError('values in indx falls outside of system')

		extra_ops = set(opstr) - self._allowed_ops
		if extra_ops:
			raise ValueError("unrecognized characters {} in operator string.".format(extra_ops))


		if self._Ns <= 0:
			return [],[],[]

		if self._unique_me:
			N_op = self.Ns
		else:
			N_op = 2*self.Ns

		col = _np.zeros(N_op,dtype=self._basis_type)
		row = _np.zeros(N_op,dtype=self._basis_type)
		ME = _np.zeros(N_op,dtype=dtype)
		error = self._op(row,col,ME,opstr,indx,J,*self._op_args,**self._blocks_1d)

		if error != 0: raise OpstrError(_basis_op_errors[error])
		mask = _np.logical_not(_np.logical_or(_np.isnan(ME),_np.abs(ME)==0.0))
		col = col[mask]
		row = row[mask]
		ME = ME[mask]

		return ME,row,col		

	def get_vec(self,v0,sparse=True):
		"""Transforms state from symmetry-reduced basis to full (symmetry-free) basis.

		Notes
		-----
		Particularly useful when a given operation canot be carried away in the symmetry-reduced basis
		in a straightforward manner.

		Supports parallelisation to multiple states listed in the columns.

		Parameters
		-----------
		v0 : numpy.ndarray
			Contains in its columns the states in the symmetry-reduced basis.
		sparse : bool, optional
			Whether or not the output should be in sparse format. Default is `True`.
		
		Returns
		--------
		numpy.ndarray
			Array containing the state `v0` in the full basis.

		Examples
		--------

		>>> v_full = get_vec(v0)
		>>> print(v_full.shape, v0.shape)

		"""

		if not hasattr(v0,"shape"):
			v0 = _np.asanyarray(v0)

		squeeze = False

		if v0.ndim == 1:
			shape = (self.sps**self.N,1)
			v0 = v0.reshape((-1,1))
			squeeze = True
		elif v0.ndim == 2:
			shape = (self.sps**self.N,v0.shape[1])
		else:
			raise ValueError("excpecting v0 to have ndim at most 2")

		if self._Ns <= 0:
			if sparse:
				return _sp.csr_matrix(([],([],[])),shape=(self.sps**self.N,0),dtype=v0.dtype)
			else:
				return _np.zeros((self.sps**self.N,0),dtype=v0.dtype)

		if v0.shape[0] != self._Ns:
			raise ValueError("v0 shape {0} not compatible with Ns={1}".format(v0.shape,self._Ns))

		if _sp.issparse(v0): # current work around for sparse states.
			return self.get_proj(v0.dtype).dot(v0)

		norms = self._get_norms(v0.dtype)

		a = self._blocks_1d.get("a")
		kblock = self._blocks_1d.get("kblock")
		pblock = self._blocks_1d.get("pblock")
		zblock = self._blocks_1d.get("zblock")
		zAblock = self._blocks_1d.get("zAblock")
		zBblock = self._blocks_1d.get("zBblock")
		pzblock = self._blocks_1d.get("pzblock")


		if (type(kblock) is int) and ((type(pblock) is int) or (type(pzblock) is int)):
			mask = (self._N < 0)
			ind_neg, = _np.nonzero(mask)
			mask = (self._N > 0)
			ind_pos, = _np.nonzero(mask)
			del mask
			def C(r,k,c,norms,dtype,ind_neg,ind_pos):
				c[ind_pos] = cos(dtype(k*r))
				c[ind_neg] = -sin(dtype(k*r))
				_np.true_divide(c,norms,c)
		else:
			ind_pos = _np.fromiter(range(v0.shape[0]),count=v0.shape[0],dtype=_np.int32)
			ind_neg = array([],dtype=_np.int32)
			def C(r,k,c,norms,dtype,*args):
				if k == 0.0:
					c[:] = 1.0
				elif k == pi:
					c[:] = (-1.0)**r
				else:
					c[:] = exp(dtype(1.0j*k*r))
				_np.true_divide(c,norms,c)

		if sparse:
			return _get_vec_sparse(self._bitops,self._pars,v0,self._basis,norms,ind_neg,ind_pos,shape,C,self._L,**self._blocks_1d)
		else:
			if squeeze:
				return  _np.squeeze(_get_vec_dense(self._bitops,self._pars,v0,self._basis,norms,ind_neg,ind_pos,shape,C,self._L,**self._blocks_1d))
			else:
				return _get_vec_dense(self._bitops,self._pars,v0,self._basis,norms,ind_neg,ind_pos,shape,C,self._L,**self._blocks_1d)

	def get_proj(self,dtype,pcon=False):
		"""Calculates transformation/projector from symmetry-reduced basis to full (symmetry-free) basis.

		Notes
		-----
		Particularly useful when a given operation canot be carried away in the symmetry-reduced basis
		in a straightforward manner.

		Parameters
		-----------
		dtype : 'type'
			Data type (e.g. numpy.float64) to construct the projector with.
		sparse : bool, optional
			Whether or not the output should be in sparse format. Default is `True`.
		pcon : bool, optional
			Whether or not to return the projector to the particle number (magnetisation) conserving basis 
			(useful in bosonic/single particle systems). Default is `pcon=False`.
		
		Returns
		--------
		scipy.sparse.csr_matrix
			Transformation/projector between the symmetry-reduced and the full basis.

		Examples
		--------

		>>> P = get_proj(np.float64,pcon=False)
		>>> print(P.shape)

		"""

		norms = self._get_norms(dtype)

		a = self._blocks_1d.get("a")
		kblock = self._blocks_1d.get("kblock")
		pblock = self._blocks_1d.get("pblock")
		zblock = self._blocks_1d.get("zblock")
		zAblock = self._blocks_1d.get("zAblock")
		zBblock = self._blocks_1d.get("zBblock")
		pzblock = self._blocks_1d.get("pzblock")

		

		if pcon and self._get_proj_pcon:
			basis_pcon = _np.ones(self._Ns_pcon,dtype=self._basis_type)
			self._make_n_basis(self.L,self._Nps,self._pars,basis_pcon)
			shape = (self._Ns_pcon,self._Ns)
		elif pcon and not self._get_proj_pcon:
			raise TypeError("pcon=True only works for basis of a single particle number sector.")
		else:
			shape = (self.sps**self.N,self._Ns)
			basis_pcon = None

		if self._Ns <= 0:
			return _sp.csr_matrix(([],([],[])),shape=shape)


		if (type(kblock) is int) and ((type(pblock) is int) or (type(pzblock) is int)):
			mask = (self._N < 0)
			ind_neg, = _np.nonzero(mask)
			mask = (self._N > 0)
			ind_pos, = _np.nonzero(mask)
			del mask
			def C(r,k,c,norms,dtype,ind_neg,ind_pos):
				c[ind_pos] = cos(dtype(k*r))
				c[ind_neg] = -sin(dtype(k*r))
				_np.true_divide(c,norms,c)
		else:
			if (type(kblock) is int):
				if ((2*kblock*a) % self._L != 0) and not _np.iscomplexobj(dtype(1.0)):
					raise TypeError("symmetries give complex vector, requested dtype is not complex")

			ind_pos = _np.arange(0,self._Ns,1)
			ind_neg = array([],dtype=_np.int32)
			def C(r,k,c,norms,dtype,*args):
				if k == 0.0:
					c[:] = 1.0
				elif k == pi:
					c[:] = (-1.0)**r
				else:
					c[:] = exp(dtype(1.0j*k*r))
				_np.true_divide(c,norms,c)





		return _get_proj_sparse(self._bitops,self._pars,self._basis,basis_pcon,norms,ind_neg,ind_pos,dtype,shape,C,self._L,**self._blocks_1d)

	def _get_norms(self,dtype):
		a = self._blocks_1d.get("a")
		kblock = self._blocks_1d.get("kblock")
		pblock = self._blocks_1d.get("pblock")
		zblock = self._blocks_1d.get("zblock")
		zAblock = self._blocks_1d.get("zAblock")
		zBblock = self._blocks_1d.get("zBblock")
		pzblock = self._blocks_1d.get("pzblock")

		if (type(kblock) is int) and (type(pblock) is int) and (type(zblock) is int):
			c = _np.empty(self._M.shape,dtype=_np.int8)
			nn = array(c)
			mm = array(c)
			sign = array(c)

			_np.sign(self._N,out=sign)			
			
			_np.floor_divide(self._M,(self._L+1)**2,out=c)
			_np.floor_divide(self._M,self._L+1,out=nn)
			_np.mod(nn,self._L+1,out=nn)
			_np.mod(self._M,self._L+1,out=mm)
			
			if _np.abs(_np.sin(self._k)) < 1.0/self._L:
				norm = _np.full(self._basis.shape,4*(self._L/a)**2,dtype=dtype)
			else:
				norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)

				
			norm *= sign
			norm /= self._N

			mask = c == 1
			norm[mask] *= (1.0 - sign[mask]*pblock*_np.cos(self._k*mm[mask]))
			_np.equal(c,2,out=mask)
			norm[mask] *= (1.0 + sign[mask]*pblock*_np.cos(self._k*mm[mask]))
			_np.equal(c,3,out=mask)
			norm[mask] *= (1.0 - zblock*_np.cos(self._k*nn[mask]))	
			_np.equal(c,4,out=mask)
			norm[mask] *= (1.0 + zblock*_np.cos(self._k*nn[mask]))	
			_np.equal(c,5,out=mask)
			norm[mask] *= (1.0 - sign[mask]*pzblock*_np.cos(self._k*mm[mask]))	
			_np.equal(c,6,out=mask)
			norm[mask] *= (1.0 + sign[mask]*pzblock*_np.cos(self._k*mm[mask]))	
			_np.equal(c,7,out=mask)
			norm[mask] *= (1.0 - sign[mask]*pblock*_np.cos(self._k*mm[mask]))
			norm[mask] *= (1.0 - zblock*_np.cos(self._k*nn[mask]))	
			_np.equal(c,8,out=mask)
			norm[mask] *= (1.0 + sign[mask]*pblock*_np.cos(self._k*mm[mask]))
			norm[mask] *= (1.0 - zblock*_np.cos(self._k*nn[mask]))	
			_np.equal(c,9,out=mask)
			norm[mask] *= (1.0 - sign[mask]*pblock*_np.cos(self._k*mm[mask]))
			norm[mask] *= (1.0 + zblock*_np.cos(self._k*nn[mask]))	
			_np.equal(c,10,out=mask)
			norm[mask] *= (1.0 + sign[mask]*pblock*_np.cos(self._k*mm[mask]))
			norm[mask] *= (1.0 + zblock*_np.cos(self._k*nn[mask]))	
			del mask
		elif (type(kblock) is int) and (type(zAblock) is int) and (type(zBblock) is int):
			c = _np.empty(self._M.shape,dtype=_np.int8)
			mm = array(c)
			_np.floor_divide(self._M,(self._L+1),c)
			_np.mod(self._M,self._L+1,mm)
			norm = _np.full(self._basis.shape,4*(self._L/a)**2,dtype=dtype)
			norm /= self._N
			mask = (c == 2)
			norm[mask] *= (1.0 + zAblock*_np.cos(self._k*mm[mask]))
			_np.equal(c,3,out=mask)
			norm[mask] *= (1.0 + zBblock*_np.cos(self._k*mm[mask]))	
			_np.equal(c,4,out=mask)
			norm[mask] *= (1.0 + zblock*_np.cos(self._k*mm[mask]))	
			del mask
		elif (type(kblock) is int) and (type(pblock) is int):
			if _np.abs(_np.sin(self._k)) < 1.0/self._L:
				norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			else:
				norm = _np.full(self._basis.shape,(self._L/a)**2,dtype=dtype)
			norm *= _np.sign(self._N)
			norm /= self._N
			try:
				m = self._M.astype(_np.min_scalar_type(-1*(self._M[0]+1)))
			except IndexError:
				m = self._M.astype(_np.int8)
			_np.mod(m,self._L+1,out=m)
			m -= 1
			mask = (m >= 0)
			sign = _np.empty(mask.sum(),dtype=_np.int8)
			_np.floor_divide(self._M[mask],(self._L+1),out=sign)
			sign *= 2
			sign -= 1
			sign *= self._N[mask]
			_np.sign(sign,out=sign)
			norm[mask] *= (1.0 + sign*pblock*_np.cos(self._k*m[mask]))
			del mask
		elif (type(kblock) is int) and (type(pzblock) is int):
			if _np.abs(_np.sin(self._k)) < 1.0/self._L:
				norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			else:
				norm = _np.full(self._basis.shape,(self._L/a)**2,dtype=dtype)
			norm *= _np.sign(self._N)
			norm /= self._N
			m = self._M.astype(_np.int8)
			_np.mod(m,self._L+1,out=m)
			m -= 1
			mask = (m >= 0)

			sign = _np.empty(mask.sum(),dtype=_np.int8)
			_np.floor_divide(self._M[mask],(self._L+1),out=sign)
			sign *= 2
			sign -= 1
			sign *= self._N[mask]
			_np.sign(sign,out=sign)

			norm[mask] *= (1.0 + sign*pzblock*_np.cos(self._k*m[mask]))
			del mask
		elif (type(kblock) is int) and (type(zblock) is int):
			norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			norm /= self._N
			
			m = self._M.astype(_np.int8)
			_np.mod(m,self._L+1,out=m)
			m -= 1
			mask = (m >= 0)

			sign = _np.empty(mask.sum(),dtype=_np.int8)
			_np.floor_divide(self._M[mask],(self._L+1),out=sign)
			sign *= 2
			sign -= 1

			norm[mask] *= (1.0 + sign*zblock*_np.cos(self._k*m[mask]))
			del mask
		elif (type(kblock) is int) and (type(zAblock) is int):
			norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			norm /= self._N
			mask = (self._M > 0)
			m = _np.empty(self._M.shape,dtype=_np.int8)
			_np.subtract(self._M,1,out=m)
			norm[mask] *= (1.0 + zAblock*_np.cos(self._k*m[mask]))
			del mask
		elif (type(kblock) is int) and (type(zBblock) is int):
			norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			norm /= self._N
			mask = (self._M > 0)
			m = _np.empty(self._M.shape,dtype=_np.int8)
			_np.subtract(self._M,1,out=m)
			norm[mask] *= (1.0 + zBblock*_np.cos(self._k*m[mask]))
			del mask
		elif (type(pblock) is int) and (type(zblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(zAblock) is int) and (type(zBblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(pblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(pzblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(zblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(zAblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(zBblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(kblock) is int):
			norm = _np.full(self._basis.shape,(self._L/a)**2,dtype=dtype)
			norm /= self._N
		else:
			norm = _np.ones(self._basis.shape,dtype=dtype)

		_np.sqrt(norm,norm)

		return norm

	##### provate methods

	def _check_symm(self,static,dynamic,photon_basis=None):
		kblock = self._blocks_1d.get("kblock")
		pblock = self._blocks_1d.get("pblock")
		zblock = self._blocks_1d.get("zblock")
		pzblock = self._blocks_1d.get("pzblock")
		zAblock = self._blocks_1d.get("zAblock")
		zBblock = self._blocks_1d.get("zBblock")
		a = self._blocks_1d.get("a")
		L = self.L

		if photon_basis is None:
			basis_sort_opstr = self._sort_opstr
			static_list,dynamic_list = self._get_local_lists(static,dynamic)
		else:
			basis_sort_opstr = photon_basis._sort_opstr
			static_list,dynamic_list = photon_basis._get_local_lists(static,dynamic)


		static_blocks = {}
		dynamic_blocks = {}
		if kblock is not None:
			missingops = _check.check_T(basis_sort_opstr,static_list,L,a)
			if missingops:	static_blocks["T symm"] = (tuple(missingops),)

			missingops = _check.check_T(basis_sort_opstr,dynamic_list,L,a)
			if missingops:	dynamic_blocks["T symm"] = (tuple(missingops),)

		if pblock is not None:
			missingops = _check.check_P(basis_sort_opstr,static_list,L)
			if missingops:	static_blocks["P symm"] = (tuple(missingops),)

			missingops = _check.check_P(basis_sort_opstr,dynamic_list,L)
			if missingops:	dynamic_blocks["P symm"] = (tuple(missingops),)

		if zblock is not None:
			oddops,missingops = _check.check_Z(basis_sort_opstr,static_list)
			if missingops or oddops:
				static_blocks["Z/C symm"] = (tuple(oddops),tuple(missingops))

			oddops,missingops = _check.check_Z(basis_sort_opstr,dynamic_list)
			if missingops or oddops:
				dynamic_blocks["Z/C symm"] = (tuple(oddops),tuple(missingops))

		if zAblock is not None:
			oddops,missingops = _check.check_ZA(basis_sort_opstr,static_list)
			if missingops or oddops:
				static_blocks["ZA/CA symm"] = (tuple(oddops),tuple(missingops))

			oddops,missingops = _check.check_ZA(basis_sort_opstr,dynamic_list)
			if missingops or oddops:
				dynamic_blocks["ZA/CA symm"] = (tuple(oddops),tuple(missingops))

		if zBblock is not None:
			oddops,missingops = _check.check_ZB(basis_sort_opstr,static_list)
			if missingops or oddops:
				static_blocks["ZB/CB symm"] = (tuple(oddops),tuple(missingops))

			oddops,missingops = _check.check_ZB(basis_sort_opstr,dynamic_list)
			if missingops or oddops:
				dynamic_blocks["ZB/CB symm"] = (tuple(oddops),tuple(missingops))

		if pzblock is not None:
			missingops = _check.check_PZ(basis_sort_opstr,static_list,L)
			if missingops:	static_blocks["PZ/PC symm"] = (tuple(missingops),)

			missingops = _check.check_PZ(basis_sort_opstr,dynamic_list,L)
			if missingops:	dynamic_blocks["PZ/PC symm"] = (tuple(missingops),)

		return static_blocks,dynamic_blocks

'''
class basis_1d(lattice_basis):
	def __init__(self,basis_module,ops_module,L,Np=None,_Np=None,pars=None,**blocks):

		if self.__class__.__name__ == "basis_1d":
			raise ValueError("This class is not intended"
							 " to be instantiated directly.")

		if type(Np) is int:
			self._check_pcon=True
			self._get_proj_pcon = True
			self._make_Np_block(basis_module,ops_module,L,Np=Np,pars=pars,**blocks)
	
		elif Np is None: # User hasn't specified Np,
			if _Np is not None: # check to see if photon_basis can create the particle sectors.

				if type(_Np) is not int:
					raise ValueError("_Np must be integer")

				if _Np == -1: 
					self._make_Np_block(basis_module,ops_module,L,pars=pars,**blocks)
				elif _Np >= 0:
					if _Np+1 > L: _Np = L
					blocks["count_particles"] = True

					zblock = blocks.get("zblock")
					zAblock = blocks.get("zAblock")
					zBblock = blocks.get("zAblock")
				
					if (type(zblock) is int) or (type(zAblock) is int) or (type(zBblock) is int):
						raise ValueError("spin inversion symmetry not compatible with particle conserving photon_basis.")
					
					# loop over the first Np particle sectors (use the iterator initialization).
					Np = list(range(_Np+1))
					self.__init__(L,Np,**blocks)
				else:
					raise ValueError("_Np == -1 for no particle conservation, _Np >= 0 for particle conservation")

			else: # if _Np is None then assume user wants to not specify Magnetization sector
				self._check_pcon = False
				self._get_proj_pcon = False
				self._make_Np_block(basis_module,ops_module,L,pars=pars,**blocks)


		else: # try to interate over Np 
			try:
				Nup_iter = iter(Np)
			except TypeError:
				raise TypeError("Np must be integer or iteratable object.")


			warnings.warn("Test for particle conservation not implemented for particle conserving lists",UserWarning,stacklevel=3)

			blocks["check_z_symm"] = False
			Np = next(Nup_iter)
			self._check_pcon = False
			self._get_proj_pcon = False
			self._make_Np_block(basis_module,ops_module,L,Np=Np,pars=pars,**blocks)
			for Np in Nup_iter:
				temp_basis =self.__class__(L,Np,**blocks)
				self._append(temp_basis)	

	def _make_Np_block(self,basis_module,ops_module,L,Np=None,pars=None,**blocks):
		# getting arguments which are used in basis.
		kblock=blocks.get("kblock")
		zblock=blocks.get("zblock")
		zAblock=blocks.get("zAblock")
		zBblock=blocks.get("zBblock")
		pblock=blocks.get("pblock")
		pzblock=blocks.get("pzblock")
		a=blocks.get("a")

		count_particles = blocks.get("count_particles")
		if count_particles is None:
			count_particles=False

		check_z_symm = blocks.get("check_z_symm")
		if check_z_symm is None:
			check_z_symm=True


		if type(L) is not int:
			raise TypeError('L must be integer')

		if self.sps < 2:
			raise ValueError("invalid value for sps, sps >= 2.")


		if type(a) is not int:
			raise TypeError('a must be integer')

		# checking if a is compatible with L
		if(L%a != 0):
			raise ValueError('L must be interger multiple of lattice spacing a')

		# checking type, and value of blocks
		if Np is not None:
			if type(Np) is not int: raise TypeError('Nup/Nb/Nf must be integer')
			if Np < 0 or Np > L*(self.sps-1): raise ValueError("0 <= Number of particles <= %d" % (L*(self.sps-1)))

		if pblock is not None:
			if type(pblock) is not int: raise TypeError('pblock must be integer')
			if abs(pblock) != 1: raise ValueError("pblock must be +/- 1")

		if zblock is not None:
			if type(zblock) is not int: raise TypeError('zblock/sblock must be integer')
			if abs(zblock) != 1: raise ValueError("zblock/sblock must be +/- 1")

		if zAblock is not None:
			if type(zAblock) is not int: raise TypeError('zAblock must be integer')
			if abs(zAblock) != 1: raise ValueError("zAblock must be +/- 1")

		if zBblock is not None:
			if type(zBblock) is not int: raise TypeError('zBblock must be integer')
			if abs(zBblock) != 1: raise ValueError("zBblock must be +/- 1")

		if pzblock is not None:
			if type(pzblock) is not int: raise TypeError('pzblock/psblock must be integer')
			if abs(pzblock) != 1: raise ValueError("pzblock/psblock must be +/- 1")

		if kblock is not None and (a <= L):
			if type(kblock) is not int: raise TypeError('kblock must be integer')
			if a == L:
				warnings.warn("using momentum with L == a",stacklevel=5)
			kblock = kblock % (L//a)
			blocks["kblock"] = kblock
			self._k = 2*pi*a*kblock/L

		self._L = L
		Ns = basis_module.get_Ns(L,Np,self.sps,**blocks) # estimate how many states in H-space to preallocate memory.
		self._basis_type = basis_module.get_basis_type(L,Np,self.sps,**blocks) # get the size of the integer representation needed for this basis (uint32,uint64,object)
		self._pars = _np.asarray(pars,dtype=self._basis_type)
		self._bitops = bitops(basis_module,**blocks)

		if type(Np) is int:
			self._conserved = "N"
			self._Ns_pcon = basis_module.get_Ns(L,Np,self.sps,**{})
			self._Np = Np
			self._make_n_basis = basis_module.n_basis
		else:
			self._conserved = ""
			self._Ns_pcon = None


		# shout out if pblock and zA/zB blocks defined simultaneously
		if type(pblock) is int and ((type(zAblock) is int) or (type(zBblock) is int)):
			raise ValueError("zA and zB symmetries incompatible with parity symmetry")

		if check_z_symm:
			blocks["Np"] = Np
			# checking if spin inversion is compatible with Np and L
			if (type(Np) is int) and ((type(zblock) is int) or (type(pzblock) is int)):
				if (L*(self.sps-1) % 2) != 0:
					raise ValueError("spin inversion/particle-hole symmetry with particle/magnetization conservation must be used with chains with 0 magnetization sector or at half filling")
				if Np != L*(self.sps-1)//2:
					raise ValueError("spin inversion/particle-hole symmetry only reduces the 0 magnetization or half filled particle sector")

			if (type(Np) is int) and ((type(zAblock) is int) or (type(zBblock) is int)):
				raise ValueError("zA/cA and zB/cB symmetries incompatible with magnetisation/particle symmetry")

			# checking if ZA/ZB spin inversion is compatible with unit cell of translation symemtry
			if (type(kblock) is int) and ((type(zAblock) is int) or (type(zBblock) is int)):
				if a%2 != 0: # T and ZA (ZB) symemtries do NOT commute
					raise ValueError("unit cell size 'a' must be even")




		self._blocks_1d = blocks
		self._unique_me = True	
		
		if (type(kblock) is int) and (type(pblock) is int) and (type(zblock) is int):
			if self._conserved: self._conserved += " & T & P & Z"
			else: self._conserved = "T & P & Z"

			self._blocks_1d["pzblock"] = pblock*zblock
			self._unique_me = False

			basis=_np.empty((Ns,),dtype=self._basis_type)
			self._op = ops_module.t_p_z_op

			if self._basis_type == _np.object:
				# if object is basis type then most likely this is for single particle stuff in which case the 
				# normalizations need to be large ~ 1000 or more which won't fit in int8/int16.
				N=_np.empty(basis.shape,dtype=_np.int32) 
				M=_np.empty(basis.shape,dtype=_np.uint32)
			else:
				N=_np.empty(basis.shape,dtype=_np.int8) # normalisation*sigma
				M=_np.empty(basis.shape,dtype=_np.uint16) # m = mp + (L+1)mz + (L+1)^2c; Anders' paper

			if (type(Np) is int):
				# arguments get overwritten by ops.-_basis 
				Ns = basis_module.n_t_p_z_basis(L,Np,pblock,zblock,kblock,a,self._pars,N,M,basis)
			else:
				Ns = basis_module.t_p_z_basis(L,pblock,zblock,kblock,a,self._pars,N,M,basis)

			# cut off extra memory for overestimated state number and reverse order
			if Ns > 0:
				self._N = N[Ns-1::-1].copy()
				self._M = M[Ns-1::-1].copy()
				self._basis = basis[Ns-1::-1].copy()
			else:
				self._N = _np.array([],dtype=N.dtype)
				self._M = _np.array([],dtype=M.dtype)
				self._basis = _np.array([],dtype=basis.dtype)
				
			self._Ns = Ns
			self._op_args=[self._N,self._M,self._basis,self._L,self._pars]

		elif (type(kblock) is int) and (type(zAblock) is int) and (type(zBblock) is int):
			if self._conserved: self._conserved += " & T & ZA & ZB"
			else: self._conserved = "T & ZA & ZB"
			self._blocks_1d["zblock"] = zAblock*zBblock


			basis=_np.empty((Ns,),dtype=self._basis_type)
			self._op = ops_module.t_zA_zB_op

			if self._basis_type == _np.object:
				N=_np.empty(basis.shape,dtype=_np.int32) 
				M=_np.empty(basis.shape,dtype=_np.uint32)
			else:
				N=_np.empty(basis.shape,dtype=_np.int8)
				M=_np.empty(basis.shape,dtype=_np.uint16)

			if (type(Np) is int):
				Ns = basis_module.n_t_zA_zB_basis(L,Np,zAblock,zBblock,kblock,a,self._pars,N,M,basis)
			else:
				Ns = basis_module.t_zA_zB_basis(L,zAblock,zBblock,kblock,a,self._pars,N,M,basis)

			if Ns > 0:
				self._N = N[Ns-1::-1].copy()
				self._M = M[Ns-1::-1].copy()
				self._basis = basis[Ns-1::-1].copy()
			else:
				self._N = _np.array([],dtype=N.dtype)
				self._M = _np.array([],dtype=M.dtype)
				self._basis = _np.array([],dtype=basis.dtype)
				
			self._Ns = Ns
			self._op_args=[self._N,self._M,self._basis,self._L,self._pars]

		elif (type(kblock) is int) and (type(pzblock) is int):
			if self._conserved: self._conserved += " & T & PZ"
			else: self._conserved = "T & PZ"
			self._unique_me = False

			basis=_np.empty((Ns,),dtype=self._basis_type)
			self._op = ops_module.t_pz_op

			if self._basis_type == _np.object:
				N=_np.empty(basis.shape,dtype=_np.int32) 
				M=_np.empty(basis.shape,dtype=_np.uint32)
			else:			
				N=_np.empty(basis.shape,dtype=_np.int8)
				M=_np.empty(basis.shape,dtype=_np.uint8) #mpz

			if (type(Np) is int):
				Ns = basis_module.n_t_pz_basis(L,Np,pzblock,kblock,a,self._pars,N,M,basis)
			else:
				Ns = basis_module.t_pz_basis(L,pzblock,kblock,a,self._pars,N,M,basis)

			if Ns > 0:
				self._N = N[Ns-1::-1].copy()
				self._M = M[Ns-1::-1].copy()
				self._basis = basis[Ns-1::-1].copy()
			else:
				self._N = _np.array([],dtype=N.dtype)
				self._M = _np.array([],dtype=M.dtype)
				self._basis = _np.array([],dtype=basis.dtype)
				
			self._Ns = Ns
			self._op_args=[self._N,self._M,self._basis,self._L,self._pars]

		elif (type(kblock) is int) and (type(pblock) is int):
			if self._conserved: self._conserved += " & T & P"
			else: self._conserved = "T & P"
			self._unique_me = False

			basis=_np.empty((Ns,),dtype=self._basis_type)
			self._op = ops_module.t_p_op

			if self._basis_type == _np.object:
				N=_np.empty(basis.shape,dtype=_np.int32) 
				M=_np.empty(basis.shape,dtype=_np.uint32)
			else:			
				N=_np.empty(basis.shape,dtype=_np.int8)
				M=_np.empty(basis.shape,dtype=_np.uint8)

			if (type(Np) is int):
				Ns = basis_module.n_t_p_basis(L,Np,pblock,kblock,a,self._pars,N,M,basis)
			else:
				Ns = basis_module.t_p_basis(L,pblock,kblock,a,self._pars,N,M,basis)

			if Ns > 0:
				self._N = N[Ns-1::-1].copy()
				self._M = M[Ns-1::-1].copy()
				self._basis = basis[Ns-1::-1].copy()
			else:
				self._N = _np.array([],dtype=N.dtype)
				self._M = _np.array([],dtype=M.dtype)
				self._basis = _np.array([],dtype=basis.dtype)
			self._Ns = Ns
			self._op_args=[self._N,self._M,self._basis,self._L,self._pars]

		elif (type(kblock) is int) and (type(zblock) is int):
			if self._conserved: self._conserved += " & T & Z"
			else: self._conserved = "T & Z"
			basis=_np.empty((Ns,),dtype=self._basis_type)
			self._op = ops_module.t_z_op

			if self._basis_type == _np.object:
				N=_np.empty(basis.shape,dtype=_np.int32) 
				M=_np.empty(basis.shape,dtype=_np.uint32)
			else:
				N=_np.empty(basis.shape,dtype=_np.int8)
				M=_np.empty(basis.shape,dtype=_np.uint8)

			if (type(Np) is int):
				Ns = basis_module.n_t_z_basis(L,Np,zblock,kblock,a,self._pars,N,M,basis)
			else:
				Ns = basis_module.t_z_basis(L,zblock,kblock,a,self._pars,N,M,basis)

			if Ns > 0:
				self._N = N[Ns-1::-1].copy()
				self._M = M[Ns-1::-1].copy()
				self._basis = basis[Ns-1::-1].copy()
			else:
				self._N = _np.array([],dtype=N.dtype)
				self._M = _np.array([],dtype=M.dtype)
				self._basis = _np.array([],dtype=basis.dtype)
				
			self._Ns = Ns
			self._op_args=[self._N,self._M,self._basis,self._L,self._pars]


		elif (type(kblock) is int) and (type(zAblock) is int):
			if self._conserved: self._conserved += " & T & ZA"
			else: self._conserved = "T & ZA"
			basis=_np.empty((Ns,),dtype=self._basis_type)
			self._op = ops_module.t_zA_op

			if self._basis_type == _np.object:
				N=_np.empty(basis.shape,dtype=_np.int32) 
				M=_np.empty(basis.shape,dtype=_np.uint32)
			else:			
				N=_np.empty(basis.shape,dtype=_np.int8)
				M=_np.empty(basis.shape,dtype=_np.uint8)

			if (type(Np) is int):
				Ns = basis_module.n_t_zA_basis(L,Np,zAblock,kblock,a,self._pars,N,M,basis)
			else:
				Ns = basis_module.t_zA_basis(L,zAblock,kblock,a,self._pars,N,M,basis)

			if Ns > 0:
				self._N = N[Ns-1::-1].copy()
				self._M = M[Ns-1::-1].copy()
				self._basis = basis[Ns-1::-1].copy()
			else:
				self._N = _np.array([],dtype=N.dtype)
				self._M = _np.array([],dtype=M.dtype)
				self._basis = _np.array([],dtype=basis.dtype)
				
			self._Ns = Ns
			self._op_args=[self._N,self._M,self._basis,self._L,self._pars]

		elif (type(kblock) is int) and (type(zBblock) is int):
			if self._conserved: self._conserved += " & T & ZB"
			else: self._conserved = "T & ZB"
			basis=_np.empty((Ns,),dtype=self._basis_type)
			self._op = ops_module.t_zB_op

			if self._basis_type == _np.object:
				N=_np.empty(basis.shape,dtype=_np.int32) 
				M=_np.empty(basis.shape,dtype=_np.uint32)
			else:			
				N=_np.empty(basis.shape,dtype=_np.int8)
				M=_np.empty(basis.shape,dtype=_np.uint8)

			if (type(Np) is int):
				Ns = basis_module.n_t_zB_basis(L,Np,zBblock,kblock,a,self._pars,N,M,basis)
			else:
				Ns = basis_module.t_zB_basis(L,zBblock,kblock,a,self._pars,N,M,basis)

			if Ns > 0:
				self._N = N[Ns-1::-1].copy()
				self._M = M[Ns-1::-1].copy()
				self._basis = basis[Ns-1::-1].copy()
			else:
				self._N = _np.array([],dtype=N.dtype)
				self._M = _np.array([],dtype=M.dtype)
				self._basis = _np.array([],dtype=basis.dtype)
			
			self._Ns = Ns
			self._op_args=[self._N,self._M,self._basis,self._L,self._pars]

		elif (type(pblock) is int) and (type(zblock) is int):
			if self._conserved: self._conserved += " & P & Z"
			else: self._conserved += "P & Z"
			basis = _np.empty((Ns,),dtype=self._basis_type)
			N=_np.empty((Ns,),dtype=_np.int8)
			self._op = ops_module.p_z_op
			
			if (type(Np) is int):
				Ns = basis_module.n_p_z_basis(L,Np,pblock,zblock,self._pars,N,basis)
			else:
				Ns = basis_module.p_z_basis(L,pblock,zblock,self._pars,N,basis)

			if Ns > 0:
				self._N = N[Ns-1::-1].copy()
				self._basis = basis[Ns-1::-1].copy()
			else:
				self._N = _np.array([],dtype=N.dtype)
				self._basis = _np.array([],dtype=basis.dtype)
				
			self._Ns = Ns
			self._op_args=[self._N,self._basis,self._L,self._pars]


		elif (type(zAblock) is int) and (type(zBblock) is int):
			if self._conserved: self._conserved += " & ZA & ZB"
			else: self._conserved += "ZA & ZB"

			self._op = ops_module.zA_zB_op

			basis = _np.empty((Ns,),dtype=self._basis_type)
			N=_np.empty((Ns,),dtype=_np.int8)
			if (type(Np) is int):
				Ns = basis_module.n_zA_zB_basis(L,Np,zAblock,zBblock,self._pars,N,basis)
			else:
				Ns = basis_module.zA_zB_basis(L,zAblock,zBblock,self._pars,N,basis)

			if Ns > 0:
				self._N = N[Ns-1::-1].copy()
				self._basis = basis[Ns-1::-1].copy()
			else:
				self._N = _np.array([],dtype=N.dtype)
				self._basis = _np.array([],dtype=basis.dtype)
				
			self._Ns = Ns
			self._op_args=[self._N,self._basis,self._L,self._pars]



		elif type(pblock) is int:
			if self._conserved: self._conserved += " & P"
			else: self._conserved = "P"
			basis = _np.empty((Ns,),dtype=self._basis_type)
			N=_np.empty((Ns,),dtype=_np.int8)
			self._op = ops_module.p_op

			if (type(Np) is int):
				Ns = basis_module.n_p_basis(L,Np,pblock,self._pars,N,basis)
			else:
				Ns = basis_module.p_basis(L,pblock,self._pars,N,basis)
				
			if Ns > 0:
				self._N = N[Ns-1::-1].copy()
				self._basis = basis[Ns-1::-1].copy()
			else:
				self._N = _np.array([],dtype=N.dtype)
				self._basis = _np.array([],dtype=basis.dtype)
				
			self._Ns = Ns
			self._op_args=[self._N,self._basis,self._L,self._pars]

		elif type(zblock) is int:
			if self._conserved: self._conserved += " & Z"
			else: self._conserved += "Z"
			basis = _np.empty((Ns,),dtype=self._basis_type)
			N=_np.empty((Ns,),dtype=_np.int8)
			self._op = ops_module.z_op

			if (type(Np) is int):
				Ns = basis_module.n_z_basis(L,Np,zblock,self._pars,N,basis)
			else:
				Ns = basis_module.z_basis(L,zblock,self._pars,N,basis)

			if Ns > 0:
				self._N = N[Ns-1::-1].copy()
				self._basis = basis[Ns-1::-1].copy()
			else:
				self._N = _np.array([],dtype=N.dtype)
				self._basis = _np.array([],dtype=basis.dtype)
				
			self._Ns = Ns
			self._op_args=[self._N,self._basis,self._L,self._pars]


			

		elif type(zAblock) is int:
			if self._conserved: self._conserved += " & ZA"
			else: self._conserved += "ZA"
			basis = _np.empty((Ns,),dtype=self._basis_type)
			N=_np.empty((Ns,),dtype=_np.int8)
			self._op = ops_module.zA_op

			if (type(Np) is int):
				Ns = basis_module.n_zA_basis(L,Np,zAblock,self._pars,N,basis)
			else:
				Ns = basis_module.zA_basis(L,zAblock,self._pars,N,basis)

			if Ns > 0:
				self._N = N[Ns-1::-1].copy()
				self._basis = basis[Ns-1::-1].copy()
			else:
				self._N = _np.array([],dtype=N.dtype)
				self._basis = _np.array([],dtype=basis.dtype)
				
			self._Ns = Ns
			self._op_args=[self._N,self._basis,self._L,self._pars]



		elif type(zBblock) is int:
			if self._conserved: self._conserved += " & ZB"
			else: self._conserved += "ZB"
			basis = _np.empty((Ns,),dtype=self._basis_type)
			N=_np.empty((Ns,),dtype=_np.int8)
			self._op = ops_module.zB_op

			if (type(Np) is int):
				Ns = basis_module.n_zB_basis(L,Np,zBblock,self._pars,N,basis)
			else:
				Ns = basis_module.zB_basis(L,zBblock,self._pars,N,basis)


			if Ns > 0:
				self._N = N[Ns-1::-1].copy()
				self._basis = basis[Ns-1::-1].copy()
			else:
				self._N = _np.array([],dtype=N.dtype)
				self._basis = _np.array([],dtype=basis.dtype)
				
			self._Ns = Ns
			self._op_args=[self._N,self._basis,self._L,self._pars]

				
		elif type(pzblock) is int:
			if self._conserved: self._conserved += " & PZ"
			else: self._conserved += "PZ"
			basis = _np.empty((Ns,),dtype=self._basis_type)
			N=_np.empty((Ns,),dtype=_np.int8)
			self._op = ops_module.pz_op

			if (type(Np) is int):
				Ns = basis_module.n_pz_basis(L,Np,pzblock,self._pars,N,basis)
			else:
				Ns = basis_module.pz_basis(L,pzblock,self._pars,N,basis)

			if Ns > 0:
				self._N = N[Ns-1::-1].copy()
				self._basis = basis[Ns-1::-1].copy()
			else:
				self._N = _np.array([],dtype=N.dtype)
				self._basis = _np.array([],dtype=basis.dtype)
				
			self._Ns = Ns
			self._op_args=[self._N,self._basis,self._L,self._pars]
	
		elif type(kblock) is int:
			if self._conserved: self._conserved += " & T"
			else: self._conserved = "T"
			basis=_np.empty((Ns,),dtype=self._basis_type)
			self._op = ops_module.t_op
			
			if self._basis_type == _np.object:
				N=_np.empty(basis.shape,dtype=_np.int32) 
			else:			
				N=_np.empty(basis.shape,dtype=_np.int8)

			if (type(Np) is int):
				Ns = basis_module.n_t_basis(L,Np,kblock,a,self._pars,N,basis)
			else:
				Ns = basis_module.t_basis(L,kblock,a,self._pars,N,basis)

			if Ns > 0:
				self._N = N[Ns-1::-1].copy()
				self._basis = basis[Ns-1::-1].copy()
			else:
				self._N = _np.array([],dtype=N.dtype)
				self._basis = _np.array([],dtype=basis.dtype)

			self._Ns = Ns
			self._op_args=[self._N,self._basis,self._L,self._pars]

		else: 
			if type(Np) is int:
				self._op = ops_module.n_op
				basis = _np.empty((Ns,),dtype=self._basis_type)
				basis_module.n_basis(L,Np,Ns,self._pars,basis)
				self._Ns = Ns
				self._basis = basis[::-1].copy()
			else:
				self._op = ops_module.op
				self._basis = _np.arange(Ns-1,-1,-1,dtype=self._basis_type)
				self._Ns = Ns
			self._op_args=[self._basis,self._pars]


		if count_particles: self._Np_list = _np.full(basis.shape,Np,dtype=_np.int8)

	def _append(self,other):
		Ns = self._Ns + other._Ns

		if self._conserved == "" or self._conserved == "N":
			self._op_args=[self._pars]
		else:
			self._op_args=[self._L,self._pars]


		self._basis.resize((Ns,),refcheck=False)
		self._basis[self._Ns:] = other._basis[:]
		arg = self._basis.argsort()[::-1]
		self._basis = self._basis[arg].copy()
		self._op_args.insert(0,self._basis)

		if hasattr(self,"_Np_list"):
			self._Np_list.resize((Ns,),refcheck=False)
			self._Np_list[self._Ns:] = other._Np_list[:]
			self._Np_list = self._Np_list[arg].copy()

		if hasattr(self,"_M"):
			self._M.resize((Ns,),refcheck=False)
			self._M[self._Ns:] = other._M[:]
			self._M = self._M[arg].copy()
			self._op_args.insert(0,self._M)	

		if hasattr(self,"_N"):
			self._N.resize((Ns,),refcheck=False)
			self._N[self._Ns:] = other._N[:]
			self._N = self._N[arg].copy()
			self._op_args.insert(0,self._N)

		self._Ns = Ns

	@property
	def L(self):
		"""int: length of lattice."""
		return self._L

	@property
	def N(self):
		"""int: number of sites the basis is constructed with."""
		return self._L


	@property
	def description(self):
		"""str: information about `basis` object."""
		blocks = ""
		lat_space = "lattice spacing: a = {a}".format(**self._blocks)

		for symm in self._blocks:
			if symm != "a":
				blocks += symm+" = {"+symm+"}, "

		blocks = blocks.format(**self._blocks)

		if len(self._conserved) == 0:
			symm = "no symmetry"
		elif len(self._conserved) == 1:
			symm = "symmetry"
		else:
			symm = "symmetries"

		string = """1d basis for chain of L = {0} containing {5} states \n\t{1}: {2} \n\tquantum numbers: {4} \n\t{3} \n\n""".format(self._L,symm,self._conserved,lat_space,blocks,self._Ns)
		string += self.operators
		return string 



	def _Op(self,opstr,indx,J,dtype):

		indx = _np.asarray(indx,dtype=_np.int32)

		if len(opstr) != len(indx):
			raise ValueError('length of opstr does not match length of indx')

		if _np.any(indx >= self._L) or _np.any(indx < 0):
			raise ValueError('values in indx falls outside of system')

		extra_ops = set(opstr) - self._allowed_ops
		if extra_ops:
			raise ValueError("unrecognized characters {} in operator string.".format(extra_ops))


		if self._Ns <= 0:
			return [],[],[]

		if self._unique_me:
			N_op = self.Ns
		else:
			N_op = 2*self.Ns

		col = _np.zeros(N_op,dtype=self._basis_type)
		row = _np.zeros(N_op,dtype=self._basis_type)
		ME = _np.zeros(N_op,dtype=dtype)
		error = self._op(row,col,ME,opstr,indx,J,*self._op_args,**self._blocks_1d)

		if error != 0: raise OpstrError(_basis_op_errors[error])
		# print ME,row,col
		mask = _np.logical_not(_np.logical_or(_np.isnan(ME),_np.abs(ME)==0.0))
		col = col[mask]
		row = row[mask]
		ME = ME[mask]

		return ME,row,col		

	def get_vec(self,v0,sparse=True):
		"""Transforms state from symmetry-reduced basis to full (symmetry-free) basis.

		Notes
		-----
		Particularly useful when a given operation canot be carried away in the symmetry-reduced basis
		in a straightforward manner.

		Supports parallelisation to multiple states listed in the columns.

		Parameters
		-----------
		v0 : numpy.ndarray
			Contains in its columns the states in the symmetry-reduced basis.
		sparse : bool, optional
			Whether or not the output should be in sparse format. Default is `True`.
		
		Returns
		--------
		numpy.ndarray
			Array containing the state `v0` in the full basis.

		Examples
		--------

		>>> v_full = get_vec(v0)
		>>> print(v_full.shape, v0.shape)

		"""

		if not hasattr(v0,"shape"):
			v0 = _np.asanyarray(v0)

		squeeze = False

		if v0.ndim == 1:
			shape = (self.sps**self.N,1)
			v0 = v0.reshape((-1,1))
			squeeze = True
		elif v0.ndim == 2:
			shape = (self.sps**self.N,v0.shape[1])
		else:
			raise ValueError("excpecting v0 to have ndim at most 2")

		if self._Ns <= 0:
			if sparse:
				return _sp.csr_matrix(([],([],[])),shape=(self.sps**self.N,0),dtype=v0.dtype)
			else:
				return _np.zeros((self.sps**self.N,0),dtype=v0.dtype)

		if v0.shape[0] != self._Ns:
			raise ValueError("v0 shape {0} not compatible with Ns={1}".format(v0.shape,self._Ns))

		if _sp.issparse(v0): # current work around for sparse states.
			return self.get_proj(v0.dtype).dot(v0)

		norms = self._get_norms(v0.dtype)

		a = self._blocks_1d.get("a")
		kblock = self._blocks_1d.get("kblock")
		pblock = self._blocks_1d.get("pblock")
		zblock = self._blocks_1d.get("zblock")
		zAblock = self._blocks_1d.get("zAblock")
		zBblock = self._blocks_1d.get("zBblock")
		pzblock = self._blocks_1d.get("pzblock")


		if (type(kblock) is int) and ((type(pblock) is int) or (type(pzblock) is int)):
			mask = (self._N < 0)
			ind_neg, = _np.nonzero(mask)
			mask = (self._N > 0)
			ind_pos, = _np.nonzero(mask)
			del mask
			def C(r,k,c,norms,dtype,ind_neg,ind_pos):
				c[ind_pos] = cos(dtype(k*r))
				c[ind_neg] = -sin(dtype(k*r))
				_np.true_divide(c,norms,c)
		else:
			ind_pos = _np.fromiter(range(v0.shape[0]),count=v0.shape[0],dtype=_np.int32)
			ind_neg = array([],dtype=_np.int32)
			def C(r,k,c,norms,dtype,*args):
				if k == 0.0:
					c[:] = 1.0
				elif k == pi:
					c[:] = (-1.0)**r
				else:
					c[:] = exp(dtype(1.0j*k*r))
				_np.true_divide(c,norms,c)

		if sparse:
			return _get_vec_sparse(self._bitops,self._pars,v0,self._basis,norms,ind_neg,ind_pos,shape,C,self._L,**self._blocks_1d)
		else:
			if squeeze:
				return  _np.squeeze(_get_vec_dense(self._bitops,self._pars,v0,self._basis,norms,ind_neg,ind_pos,shape,C,self._L,**self._blocks_1d))
			else:
				return _get_vec_dense(self._bitops,self._pars,v0,self._basis,norms,ind_neg,ind_pos,shape,C,self._L,**self._blocks_1d)

	def get_proj(self,dtype,pcon=False):
		"""Calculates transformation/projector from symmetry-reduced basis to full (symmetry-free) basis.

		Notes
		-----
		Particularly useful when a given operation canot be carried away in the symmetry-reduced basis
		in a straightforward manner.

		Parameters
		-----------
		dtype : 'type'
			Data type (e.g. numpy.float64) to construct the projector with.
		sparse : bool, optional
			Whether or not the output should be in sparse format. Default is `True`.
		pcon : bool, optional
			Whether or not to return the projector to the particle number (magnetisation) conserving basis 
			(useful in bosonic/single particle systems). Default is `pcon=False`.
		
		Returns
		--------
		scipy.sparse.csr_matrix
			Transformation/projector between the symmetry-reduced and the full basis.

		Examples
		--------

		>>> P = get_proj(np.float64,pcon=False)
		>>> print(P.shape)

		"""

		norms = self._get_norms(dtype)

		a = self._blocks_1d.get("a")
		kblock = self._blocks_1d.get("kblock")
		pblock = self._blocks_1d.get("pblock")
		zblock = self._blocks_1d.get("zblock")
		zAblock = self._blocks_1d.get("zAblock")
		zBblock = self._blocks_1d.get("zBblock")
		pzblock = self._blocks_1d.get("pzblock")

		

		if pcon and self._get_proj_pcon:
			basis_pcon = _np.ones(self._Ns_pcon,dtype=self._basis_type)
			self._make_n_basis(self.L,self._Np,self._Ns_pcon,self._pars,basis_pcon)
			shape = (self._Ns_pcon,self._Ns)
		elif pcon and not self._get_proj_pcon:
			raise TypeError("pcon=True only works for basis of a single particle number sector.")
		else:
			shape = (self.sps**self.L,self._Ns)
			basis_pcon = None

		if self._Ns <= 0:
			return _sp.csr_matrix(([],([],[])),shape=shape)


		if (type(kblock) is int) and ((type(pblock) is int) or (type(pzblock) is int)):
			mask = (self._N < 0)
			ind_neg, = _np.nonzero(mask)
			mask = (self._N > 0)
			ind_pos, = _np.nonzero(mask)
			del mask
			def C(r,k,c,norms,dtype,ind_neg,ind_pos):
				c[ind_pos] = cos(dtype(k*r))
				c[ind_neg] = -sin(dtype(k*r))
				_np.true_divide(c,norms,c)
		else:
			if (type(kblock) is int):
				if ((2*kblock*a) % self._L != 0) and not _np.iscomplexobj(dtype(1.0)):
					raise TypeError("symmetries give complex vector, requested dtype is not complex")

			ind_pos = _np.arange(0,self._Ns,1)
			ind_neg = array([],dtype=_np.int32)
			def C(r,k,c,norms,dtype,*args):
				if k == 0.0:
					c[:] = 1.0
				elif k == pi:
					c[:] = (-1.0)**r
				else:
					c[:] = exp(dtype(1.0j*k*r))
				_np.true_divide(c,norms,c)





		return _get_proj_sparse(self._bitops,self._pars,self._basis,basis_pcon,norms,ind_neg,ind_pos,dtype,shape,C,self._L,**self._blocks_1d)

	def _get_norms(self,dtype):
		a = self._blocks_1d.get("a")
		kblock = self._blocks_1d.get("kblock")
		pblock = self._blocks_1d.get("pblock")
		zblock = self._blocks_1d.get("zblock")
		zAblock = self._blocks_1d.get("zAblock")
		zBblock = self._blocks_1d.get("zBblock")
		pzblock = self._blocks_1d.get("pzblock")

		if (type(kblock) is int) and (type(pblock) is int) and (type(zblock) is int):
			c = _np.empty(self._M.shape,dtype=_np.int8)
			nn = array(c)
			mm = array(c)
			sign = array(c)

			_np.sign(self._N,out=sign)			
			
			_np.floor_divide(self._M,(self._L+1)**2,out=c)
			_np.floor_divide(self._M,self._L+1,out=nn)
			_np.mod(nn,self._L+1,out=nn)
			_np.mod(self._M,self._L+1,out=mm)
			
			if _np.abs(_np.sin(self._k)) < 1.0/self._L:
				norm = _np.full(self._basis.shape,4*(self._L/a)**2,dtype=dtype)
			else:
				norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)

				
			norm *= sign
			norm /= self._N

			mask = c == 1
			norm[mask] *= (1.0 - sign[mask]*pblock*_np.cos(self._k*mm[mask]))
			_np.equal(c,2,out=mask)
			norm[mask] *= (1.0 + sign[mask]*pblock*_np.cos(self._k*mm[mask]))
			_np.equal(c,3,out=mask)
			norm[mask] *= (1.0 - zblock*_np.cos(self._k*nn[mask]))	
			_np.equal(c,4,out=mask)
			norm[mask] *= (1.0 + zblock*_np.cos(self._k*nn[mask]))	
			_np.equal(c,5,out=mask)
			norm[mask] *= (1.0 - sign[mask]*pzblock*_np.cos(self._k*mm[mask]))	
			_np.equal(c,6,out=mask)
			norm[mask] *= (1.0 + sign[mask]*pzblock*_np.cos(self._k*mm[mask]))	
			_np.equal(c,7,out=mask)
			norm[mask] *= (1.0 - sign[mask]*pblock*_np.cos(self._k*mm[mask]))
			norm[mask] *= (1.0 - zblock*_np.cos(self._k*nn[mask]))	
			_np.equal(c,8,out=mask)
			norm[mask] *= (1.0 + sign[mask]*pblock*_np.cos(self._k*mm[mask]))
			norm[mask] *= (1.0 - zblock*_np.cos(self._k*nn[mask]))	
			_np.equal(c,9,out=mask)
			norm[mask] *= (1.0 - sign[mask]*pblock*_np.cos(self._k*mm[mask]))
			norm[mask] *= (1.0 + zblock*_np.cos(self._k*nn[mask]))	
			_np.equal(c,10,out=mask)
			norm[mask] *= (1.0 + sign[mask]*pblock*_np.cos(self._k*mm[mask]))
			norm[mask] *= (1.0 + zblock*_np.cos(self._k*nn[mask]))	
			del mask
		elif (type(kblock) is int) and (type(zAblock) is int) and (type(zBblock) is int):
			c = _np.empty(self._M.shape,dtype=_np.int8)
			mm = array(c)
			_np.floor_divide(self._M,(self._L+1),c)
			_np.mod(self._M,self._L+1,mm)
			norm = _np.full(self._basis.shape,4*(self._L/a)**2,dtype=dtype)
			norm /= self._N
			mask = (c == 2)
			norm[mask] *= (1.0 + zAblock*_np.cos(self._k*mm[mask]))
			_np.equal(c,3,out=mask)
			norm[mask] *= (1.0 + zBblock*_np.cos(self._k*mm[mask]))	
			_np.equal(c,4,out=mask)
			norm[mask] *= (1.0 + zblock*_np.cos(self._k*mm[mask]))	
			del mask
		elif (type(kblock) is int) and (type(pblock) is int):
			if _np.abs(_np.sin(self._k)) < 1.0/self._L:
				norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			else:
				norm = _np.full(self._basis.shape,(self._L/a)**2,dtype=dtype)
			norm *= _np.sign(self._N)
			norm /= self._N
			try:
				m = self._M.astype(_np.min_scalar_type(-1*(self._M[0]+1)))
			except IndexError:
				m = self._M.astype(_np.int8)
			_np.mod(m,self._L+1,out=m)
			m -= 1
			mask = (m >= 0)
			sign = _np.empty(mask.sum(),dtype=_np.int8)
			_np.floor_divide(self._M[mask],(self._L+1),out=sign)
			sign *= 2
			sign -= 1
			sign *= self._N[mask]
			_np.sign(sign,out=sign)
			norm[mask] *= (1.0 + sign*pblock*_np.cos(self._k*m[mask]))
			del mask
		elif (type(kblock) is int) and (type(pzblock) is int):
			if _np.abs(_np.sin(self._k)) < 1.0/self._L:
				norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			else:
				norm = _np.full(self._basis.shape,(self._L/a)**2,dtype=dtype)
			norm *= _np.sign(self._N)
			norm /= self._N
			m = self._M.astype(_np.int8)
			_np.mod(m,self._L+1,out=m)
			m -= 1
			mask = (m >= 0)

			sign = _np.empty(mask.sum(),dtype=_np.int8)
			_np.floor_divide(self._M[mask],(self._L+1),out=sign)
			sign *= 2
			sign -= 1
			sign *= self._N[mask]
			_np.sign(sign,out=sign)

			norm[mask] *= (1.0 + sign*pzblock*_np.cos(self._k*m[mask]))
			del mask
		elif (type(kblock) is int) and (type(zblock) is int):
			norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			norm /= self._N
			
			m = self._M.astype(_np.int8)
			_np.mod(m,self._L+1,out=m)
			m -= 1
			mask = (m >= 0)

			sign = _np.empty(mask.sum(),dtype=_np.int8)
			_np.floor_divide(self._M[mask],(self._L+1),out=sign)
			sign *= 2
			sign -= 1

			norm[mask] *= (1.0 + sign*zblock*_np.cos(self._k*m[mask]))
			del mask
		elif (type(kblock) is int) and (type(zAblock) is int):
			norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			norm /= self._N
			mask = (self._M > 0)
			m = _np.empty(self._M.shape,dtype=_np.int8)
			_np.subtract(self._M,1,out=m)
			norm[mask] *= (1.0 + zAblock*_np.cos(self._k*m[mask]))
			del mask
		elif (type(kblock) is int) and (type(zBblock) is int):
			norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			norm /= self._N
			mask = (self._M > 0)
			m = _np.empty(self._M.shape,dtype=_np.int8)
			_np.subtract(self._M,1,out=m)
			norm[mask] *= (1.0 + zBblock*_np.cos(self._k*m[mask]))
			del mask
		elif (type(pblock) is int) and (type(zblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(zAblock) is int) and (type(zBblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(pblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(pzblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(zblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(zAblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(zBblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(kblock) is int):
			norm = _np.full(self._basis.shape,(self._L/a)**2,dtype=dtype)
			norm /= self._N
		else:
			norm = _np.ones(self._basis.shape,dtype=dtype)

		_np.sqrt(norm,norm)

		return norm

	##### provate methods

	def _check_symm(self,static,dynamic,photon_basis=None):
		kblock = self._blocks_1d.get("kblock")
		pblock = self._blocks_1d.get("pblock")
		zblock = self._blocks_1d.get("zblock")
		pzblock = self._blocks_1d.get("pzblock")
		zAblock = self._blocks_1d.get("zAblock")
		zBblock = self._blocks_1d.get("zBblock")
		a = self._blocks_1d.get("a")
		L = self.L

		if photon_basis is None:
			basis_sort_opstr = self._sort_opstr
			static_list,dynamic_list = self._get_local_lists(static,dynamic)
		else:
			basis_sort_opstr = photon_basis._sort_opstr
			static_list,dynamic_list = photon_basis._get_local_lists(static,dynamic)


		static_blocks = {}
		dynamic_blocks = {}
		if kblock is not None:
			missingops = _check.check_T(basis_sort_opstr,static_list,L,a)
			if missingops:	static_blocks["T symm"] = (tuple(missingops),)

			missingops = _check.check_T(basis_sort_opstr,dynamic_list,L,a)
			if missingops:	dynamic_blocks["T symm"] = (tuple(missingops),)

		if pblock is not None:
			missingops = _check.check_P(basis_sort_opstr,static_list,L)
			if missingops:	static_blocks["P symm"] = (tuple(missingops),)

			missingops = _check.check_P(basis_sort_opstr,dynamic_list,L)
			if missingops:	dynamic_blocks["P symm"] = (tuple(missingops),)

		if zblock is not None:
			oddops,missingops = _check.check_Z(basis_sort_opstr,static_list)
			if missingops or oddops:
				static_blocks["Z/C symm"] = (tuple(oddops),tuple(missingops))

			oddops,missingops = _check.check_Z(basis_sort_opstr,dynamic_list)
			if missingops or oddops:
				dynamic_blocks["Z/C symm"] = (tuple(oddops),tuple(missingops))

		if zAblock is not None:
			oddops,missingops = _check.check_ZA(basis_sort_opstr,static_list)
			if missingops or oddops:
				static_blocks["ZA/CA symm"] = (tuple(oddops),tuple(missingops))

			oddops,missingops = _check.check_ZA(basis_sort_opstr,dynamic_list)
			if missingops or oddops:
				dynamic_blocks["ZA/CA symm"] = (tuple(oddops),tuple(missingops))

		if zBblock is not None:
			oddops,missingops = _check.check_ZB(basis_sort_opstr,static_list)
			if missingops or oddops:
				static_blocks["ZB/CB symm"] = (tuple(oddops),tuple(missingops))

			oddops,missingops = _check.check_ZB(basis_sort_opstr,dynamic_list)
			if missingops or oddops:
				dynamic_blocks["ZB/CB symm"] = (tuple(oddops),tuple(missingops))

		if pzblock is not None:
			missingops = _check.check_PZ(basis_sort_opstr,static_list,L)
			if missingops:	static_blocks["PZ/PC symm"] = (tuple(missingops),)

			missingops = _check.check_PZ(basis_sort_opstr,dynamic_list,L)
			if missingops:	dynamic_blocks["PZ/PC symm"] = (tuple(missingops),)

		return static_blocks,dynamic_blocks
'''

def _get_vec_dense(ops,pars,v0,basis_in,norms,ind_neg,ind_pos,shape,C,L,**blocks):
	dtype=_dtypes[v0.dtype.char]

	a = blocks.get("a")
	kblock = blocks.get("kblock")
	pblock = blocks.get("pblock")
	zblock = blocks.get("zblock")
	zAblock = blocks.get("zAblock")
	zBblock = blocks.get("zBblock")
	pzblock = blocks.get("pzblock")


	c = _np.zeros(basis_in.shape,dtype=v0.dtype)
	sign = _np.ones(basis_in.shape[0],dtype=_np.int8)	
	v = _np.zeros(shape,dtype=v0.dtype)

	if type(kblock) is int:
		k = 2*pi*kblock*a/L
	else:
		k = 0.0
		a = L

	Ns_full = shape[0]
	v_rev = v[::-1] # access array in reverse order. 

	for r in range(0,L//a):
		C(r,k,c,norms,dtype,ind_neg,ind_pos)	
		vc = (v0.T*c).T
		vc_tran = vc.transpose()
		vc_tran *= sign
		v_rev[basis_in[ind_pos]] += vc[ind_pos]
		v_rev[basis_in[ind_neg]] += vc[ind_neg]
		vc_tran *= sign

		if type(zAblock) is int:
			ops.py_flip_sublat_A(basis_in,L,pars,sign)
			vc *= zAblock
			vc_tran *= sign
			v_rev[basis_in[ind_pos]] += vc[ind_pos]
			v_rev[basis_in[ind_neg]] += vc[ind_neg]
			vc *= zAblock
			vc_tran *= sign
			ops.py_flip_sublat_A(basis_in,L,pars,sign)
		
		if type(zBblock) is int:
			ops.py_flip_sublat_B(basis_in,L,pars,sign)
			vc *= zBblock
			vc_tran *= sign
			v_rev[basis_in[ind_pos]] += vc[ind_pos]
			v_rev[basis_in[ind_neg]] += vc[ind_neg]
			vc *= zBblock
			vc_tran *= sign
			ops.py_flip_sublat_B(basis_in,L,pars,sign)
		
		if type(zblock) is int:
			ops.py_flip_all(basis_in,L,pars,sign)
			vc *= zblock
			vc_tran *= sign
			v_rev[basis_in[ind_pos]] += vc[ind_pos]
			v_rev[basis_in[ind_neg]] += vc[ind_neg]
			vc *= zblock
			vc_tran *= sign
			ops.py_flip_all(basis_in,L,pars,sign)

		if type(pblock) is int:
			ops.py_fliplr(basis_in,L,pars,sign)
			vc *= pblock
			vc_tran *= sign
			v_rev[basis_in[ind_pos]] += vc[ind_pos]
			v_rev[basis_in[ind_neg]] += vc[ind_neg]
			vc *= pblock
			vc_tran *= sign
			ops.py_fliplr(basis_in,L,pars,sign)

		if type(pzblock) is int:
			ops.py_fliplr(basis_in,L,pars,sign)
			ops.py_flip_all(basis_in,L,pars,sign)
			vc *= pzblock
			vc_tran *= sign
			v_rev[basis_in[ind_pos]] += vc[ind_pos]
			v_rev[basis_in[ind_neg]] += vc[ind_neg]
			vc *= pzblock
			vc_tran *= sign
			ops.py_fliplr(basis_in,L,pars,sign)
			ops.py_flip_all(basis_in,L,pars,sign)
		
		ops.py_shift(basis_in,a,L,pars,sign)
	
	return v

def _get_vec_sparse(ops,pars,v0,basis_in,norms,ind_neg,ind_pos,shape,C,L,**blocks):
	dtype=_dtypes[v0.dtype.char]

	a = blocks.get("a")
	kblock = blocks.get("kblock")
	pblock = blocks.get("pblock")
	zblock = blocks.get("zblock")
	zAblock = blocks.get("zAblock")
	zBblock = blocks.get("zBblock")
	pzblock = blocks.get("pzblock")

	m = shape[1]


	
	if ind_neg.shape[0] == 0:
		row_neg = array([],dtype=_np.int64)
		col_neg = array([],dtype=_np.int64)
	else:
		col_neg = _np.arange(0,m,1)
		row_neg = _np.kron(ind_neg,_np.ones_like(col_neg))
		col_neg = _np.kron(_np.ones_like(ind_neg),col_neg)

	if ind_pos.shape[0] == 0:
		row_pos = array([],dtype=_np.int64)
		col_pos = array([],dtype=_np.int64)
	else:
		col_pos = _np.arange(0,m,1)
		row_pos = _np.kron(ind_pos,_np.ones_like(col_pos))
		col_pos = _np.kron(_np.ones_like(ind_pos),col_pos)


	c = _np.zeros(basis_in.shape,dtype=v0.dtype)	
	sign = _np.ones(basis_in.shape,dtype=_np.int8)
	v = _sp.csr_matrix(shape,dtype=v0.dtype)

	if type(kblock) is int:
		k = 2*pi*kblock*a/L
	else:
		k = 0.0
		a = L


	Ns_full = shape[0]
	index = _np.zeros_like(basis_in)

	for r in range(0,L//a):
		C(r,k,c,norms,dtype,ind_neg,ind_pos)
		vc = (v0.T*c).T
		data_pos = vc[ind_pos].copy()
		data_neg = vc[ind_neg].copy()
		# view which passes into sparse matrix constructor
		data_pos_flat = data_pos.reshape((-1,))
		data_neg_flat = data_neg.reshape((-1,))
		# view which us used to multiply by sign
		data_pos_tran = data_pos.transpose()
		data_neg_tran = data_neg.transpose()
		
		data_pos_tran *= sign[ind_pos]
		data_neg_tran *= sign[ind_neg]
		index[:] = Ns_full - 1
		index -= basis_in
		v = v + _sp.csr_matrix((data_pos_flat,(index[row_pos],col_pos)),shape,dtype=v.dtype)
		v = v + _sp.csr_matrix((data_neg_flat,(index[row_neg],col_neg)),shape,dtype=v.dtype)
		data_pos_tran *= sign[ind_pos]
		data_neg_tran *= sign[ind_neg]

		index[:] = Ns_full - 1
		if type(zAblock) is int:
			ops.py_flip_sublat_A(basis_in,L,pars,sign)
			data_pos *= zAblock
			data_pos_tran *= sign[ind_pos]
			data_neg *= zAblock
			data_neg_tran *= sign[ind_neg]
			index -= basis_in
			v = v + _sp.csr_matrix((data_pos_flat,(index[row_pos],col_pos)),shape,dtype=v.dtype)
			v = v + _sp.csr_matrix((data_neg_flat,(index[row_neg],col_neg)),shape,dtype=v.dtype)
			data_pos *= zAblock
			data_pos_tran *= sign[ind_pos]
			data_neg *= zAblock
			data_neg_tran *= sign[ind_neg]
			ops.py_flip_sublat_A(basis_in,L,pars,sign)

		index[:] = Ns_full - 1
		if type(zBblock) is int:
			ops.py_flip_sublat_B(basis_in,L,pars,sign)
			data_pos *= zBblock
			data_pos_tran *= sign[ind_pos]
			data_neg *= zBblock
			data_neg_tran *= sign[ind_neg]
			index -= basis_in
			v = v + _sp.csr_matrix((data_pos_flat,(index[row_pos],col_pos)),shape,dtype=v.dtype)
			v = v + _sp.csr_matrix((data_neg_flat,(index[row_neg],col_neg)),shape,dtype=v.dtype)
			data_pos *= zBblock
			data_pos_tran *= sign[ind_pos]
			data_neg *= zBblock
			data_neg_tran *= sign[ind_neg]
			ops.py_flip_sublat_B(basis_in,L,pars,sign)

		index[:] = Ns_full - 1
		if type(zblock) is int:
			ops.py_flip_all(basis_in,L,pars,sign)
			data_pos *= zblock
			data_pos_tran *= sign[ind_pos]
			data_neg *= zblock
			data_neg_tran *= sign[ind_neg]
			index -= basis_in
			v = v + _sp.csr_matrix((data_pos_flat,(index[row_pos],col_pos)),shape,dtype=v.dtype)
			v = v + _sp.csr_matrix((data_neg_flat,(index[row_neg],col_neg)),shape,dtype=v.dtype)
			data_pos *= zblock
			data_pos_tran *= sign[ind_pos]
			data_neg *= zblock
			data_neg_tran *= sign[ind_neg]
			ops.py_flip_all(basis_in,L,pars,sign)

		index[:] = Ns_full - 1
		if type(pblock) is int:
			ops.py_fliplr(basis_in,L,pars,sign)
			data_pos *= pblock
			data_pos_tran *= sign[ind_pos]
			data_neg *= pblock
			data_neg_tran *= sign[ind_neg]
			index -= basis_in
			v = v + _sp.csr_matrix((data_pos_flat,(index[row_pos],col_pos)),shape,dtype=v.dtype)
			v = v + _sp.csr_matrix((data_neg_flat,(index[row_neg],col_neg)),shape,dtype=v.dtype)
			data_pos *= pblock
			data_pos_tran *= sign[ind_pos]
			data_neg *= pblock
			data_neg_tran *= sign[ind_neg]
			ops.py_fliplr(basis_in,L,pars,sign)

		index[:] = Ns_full - 1
		if type(pzblock) is int:
			ops.py_flip_all(basis_in,L,pars,sign)
			ops.py_fliplr(basis_in,L,pars,sign)
			data_pos *= pzblock
			data_pos_tran *= sign[ind_pos]
			data_neg *= pzblock
			data_neg_tran *= sign[ind_neg]
			index -= basis_in
			v = v + _sp.csr_matrix((data_pos_flat,(index[row_pos],col_pos)),shape,dtype=v.dtype)
			v = v + _sp.csr_matrix((data_neg_flat,(index[row_neg],col_neg)),shape,dtype=v.dtype)
			data_pos *= pzblock
			data_pos_tran *= sign[ind_pos]
			data_neg *= pzblock
			data_neg_tran *= sign[ind_neg]
			ops.py_fliplr(basis_in,L,pars,sign)
			ops.py_flip_all(basis_in,L,pars,sign)

		v.sum_duplicates()
		v.eliminate_zeros()
		ops.py_shift(basis_in,a,L,pars,sign)

	return v

def _get_proj_sparse(ops,pars,basis_in,basis_pcon,norms,ind_neg,ind_pos,dtype,shape,C,L,**blocks):

	a = blocks.get("a")
	kblock = blocks.get("kblock")
	pblock = blocks.get("pblock")
	zblock = blocks.get("zblock")
	zAblock = blocks.get("zAblock")
	zBblock = blocks.get("zBblock")
	pzblock = blocks.get("pzblock")


	if type(kblock) is int:
		k = 2*pi*kblock*a/L
	else:
		k = 0.0
		a = L

	c = _np.zeros(basis_in.shape,dtype=dtype)
	sign = _np.ones(basis_in.shape,dtype=_np.int8)
	v = _sp.csr_matrix(shape,dtype=dtype)

	if basis_pcon is None:
		def get_index(ind):
			return shape[0] - basis_in[ind] - 1
	else:
		def get_index(ind):
			return shape[0] - basis_pcon.searchsorted(basis_in[ind]) - 1

	for r in range(0,L//a):
		C(r,k,c,norms,dtype,ind_neg,ind_pos)
		data_pos = c[ind_pos]
		data_neg = c[ind_neg]

		data_pos *= sign[ind_pos]
		data_neg *= sign[ind_neg]
		index = get_index(ind_pos)
		v = v + _sp.csr_matrix((data_pos,(index,ind_pos)),shape,dtype=v.dtype)
		index = get_index(ind_neg)
		v = v + _sp.csr_matrix((data_neg,(index,ind_neg)),shape,dtype=v.dtype)
		data_pos *= sign[ind_pos]
		data_neg *= sign[ind_neg]


		if type(zAblock) is int:
			ops.py_flip_sublat_A(basis_in,L,pars,sign)
			data_pos *= zAblock
			data_pos *= sign[ind_pos]
			data_neg *= zAblock
			data_neg *= sign[ind_neg]
			index = get_index(ind_pos)
			v = v + _sp.csr_matrix((data_pos,(index,ind_pos)),shape,dtype=v.dtype)
			index = get_index(ind_neg)
			v = v + _sp.csr_matrix((data_neg,(index,ind_neg)),shape,dtype=v.dtype)
			data_pos *= zAblock
			data_pos *= sign[ind_pos]
			data_neg *= zAblock
			data_neg *= sign[ind_neg]
			ops.py_flip_sublat_A(basis_in,L,pars,sign)

		if type(zBblock) is int:
			ops.py_flip_sublat_B(basis_in,L,pars,sign)
			data_pos *= zBblock
			data_pos *= sign[ind_pos]
			data_neg *= zBblock
			data_neg *= sign[ind_neg]
			index = get_index(ind_pos)
			v = v + _sp.csr_matrix((data_pos,(index,ind_pos)),shape,dtype=v.dtype)
			index = get_index(ind_neg)
			v = v + _sp.csr_matrix((data_neg,(index,ind_neg)),shape,dtype=v.dtype)
			data_pos *= zBblock
			data_pos *= sign[ind_pos]
			data_neg *= zBblock
			data_neg *= sign[ind_neg]
			ops.py_flip_sublat_B(basis_in,L,pars,sign)

		if type(zblock) is int:
			ops.py_flip_all(basis_in,L,pars,sign)
			data_pos *= zblock
			data_pos *= sign[ind_pos]
			data_neg *= zblock
			data_neg *= sign[ind_neg]
			index = get_index(ind_pos)
			v = v + _sp.csr_matrix((data_pos,(index,ind_pos)),shape,dtype=v.dtype)
			index = get_index(ind_neg)
			v = v + _sp.csr_matrix((data_neg,(index,ind_neg)),shape,dtype=v.dtype)
			data_pos *= zblock
			data_pos *= sign[ind_pos]
			data_neg *= zblock
			data_neg *= sign[ind_neg]
			ops.py_flip_all(basis_in,L,pars,sign)

		if type(pblock) is int:
			ops.py_fliplr(basis_in,L,pars,sign)
			data_pos *= pblock
			data_pos *= sign[ind_pos]
			data_neg *= pblock
			data_neg *= sign[ind_neg]
			index = get_index(ind_pos)
			v = v + _sp.csr_matrix((data_pos,(index,ind_pos)),shape,dtype=v.dtype)
			index = get_index(ind_neg)
			v = v + _sp.csr_matrix((data_neg,(index,ind_neg)),shape,dtype=v.dtype)
			data_pos *= pblock
			data_pos *= sign[ind_pos]
			data_neg *= pblock
			data_neg *= sign[ind_neg]
			ops.py_fliplr(basis_in,L,pars,sign)

		if type(pzblock) is int:
			ops.py_fliplr(basis_in,L,pars,sign)
			ops.py_flip_all(basis_in,L,pars,sign)
			data_pos *= pzblock
			data_pos *= sign[ind_pos]
			data_neg *= pzblock
			data_neg *= sign[ind_neg]
			index = get_index(ind_pos)
			v = v + _sp.csr_matrix((data_pos,(index,ind_pos)),shape,dtype=v.dtype)
			index = get_index(ind_neg)
			v = v + _sp.csr_matrix((data_neg,(index,ind_neg)),shape,dtype=v.dtype)
			data_pos *= pzblock
			data_pos *= sign[ind_pos]
			data_neg *= pzblock
			data_neg *= sign[ind_neg]
			ops.py_fliplr(basis_in,L,pars,sign)
			ops.py_flip_all(basis_in,L,pars,sign)

		ops.py_shift(basis_in,a,L,pars,sign)


	return v



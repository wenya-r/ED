from ._basis_1d_core import hcp_basis,hcp_ops
from ._basis_1d_core import boson_basis,boson_ops
from .base_1d import basis_1d
import numpy as _np

try:
	S_dict = {(str(i)+"/2" if i%2==1 else str(i//2)):(i+1,i/2.0) for i in xrange(1,10001)}
except NameError:
	S_dict = {(str(i)+"/2" if i%2==1 else str(i//2)):(i+1,i/2.0) for i in range(1,10001)}

class spin_basis_1d(basis_1d):
	"""Constructs basis for spin operators in a specified 1-d symmetry sector.

	The supported operator strings for `spin_basis_1d` are:

	.. math::
		\\begin{array}{cccc}
			\\texttt{basis}/\\texttt{opstr}   &   \\texttt{"I"}   &   \\texttt{"+"}   &   \\texttt{"-"}  &     \\texttt{"z"}   &   \\texttt{"x"}   &   \\texttt{"y"}  \\newline	
			\\texttt{spin_basis_1d} &   \\hat{1}        &   \\hat\\sigma^+       &   \\hat\\sigma^-      &     \\hat\\sigma^z       &   (\\hat\\sigma^x)     &   (\\hat\\sigma^y)  \\  \\newline
		\\end{array}

	**Note:** The default operators for spin-1/2 are the Pauli matrices, NOT the spin operators. To change this, see
	the argument `pauli` of the `spin_basis` class. Higher spins can only be defined using the spin operators, and do NOT support
	the operator strings "x" and "y". 

	Examples
	--------

	The code snippet below shows how to use the `spin_boson_1d` class to construct the basis in the zero momentum sector of positive parity for the spin Hamiltonian.

	.. math::
		H(t) = \\sum_j J\\sigma^z_{j+1}\\sigma^z_j + h\\sigma^z_j + g\\cos\\Omega t\\sigma^x_j 

	.. literalinclude:: ../../doc_examples/spin_basis_1d-example.py
		:linenos:
		:language: python
		:lines: 7-

	"""	
	def __init__(self,L,Nup=None,m=None,S="1/2",pauli=True,**blocks):
		"""Intializes the `spin_basis_1d` object (basis for spin operators).

		Parameters
		-----------
		L: int
			Length of chain/number of sites.
		Nup: {int,list}, optional
			Total magnetisation, :math:`\\sum_j S^z_j`, projection. Can be integer or list to specify one or 
			more particle sectors.
		m: float, optional
			Density of spin up in chain (spin up per site).
		S: str, optional
			Size of local spin degrees of freedom. Can be any (half-)integer from:
			"1/2","1","3/2",...,"9999/2","5000".
		pauli: bool, optional
			Whether or not to use Pauli or spin-1/2 operators. Requires `S=1/2`.
		**blocks: optional
			extra keyword arguments which include:

				**a** (*int*) - specifies unit cell size for translation.

				**kblock** (*int*) - specifies momentum block.

				**pblock** (*int*) - specifies parity block.

				**zblock** (*int*) - specifies spin inversion symmetry block.

				**pzblock** (*int*) - specifies parity followed by spin inversion symmetry block.

				**zAblock** (*int*) - specifies spin inversion symmetry block for sublattice A.

				**zBblock** (*int*) - specifies spin inversion symmetry block for sublattice B.

		"""
		input_keys = set(blocks.keys())
		expected_keys = set(["_Np","kblock","zblock","zAblock","zBblock","pblock","pzblock","a","count_particles","check_z_symm","L"])
		wrong_keys = input_keys - expected_keys 
		if wrong_keys:
			temp = ", ".join(["{}" for key in wrong_keys])
			raise ValueError(("unexpected optional argument(s): "+temp).format(*wrong_keys))


		self._sps,S = S_dict[S]
		if Nup is not None and m is not None:
			raise ValueError("Cannot use Nup and m at the same time")
		elif Nup is None and m is not None:
			if m < -S or m > S:
				raise ValueError("N must be between -S and S")

			Nup = int((m+S)*L)

		if Nup is None:
			Nup_list = None
		elif type(Nup) is int:
			Nup_list = [Nup]
		else:
			try:
				Nup_list = list(Nup)
			except TypeError:
				raise TypeError("Nup must be iterable returning integers")

			if any((type(Nup) is not int) for Nup in Nup_list):
				TypeError("Nup must be iterable returning integers")

		count_particles = False
		if blocks.get("_Np") is not None:
			_Np = blocks.get("_Np")
			if Nup_list is not None:
				raise ValueError("do not use _Np and Nup/nb simultaineously.")
			blocks.pop("_Np")
			if _Np == -1:
				Nup_list = None
			else:
				count_particles = True
				if _Np+1 > L: _Np = L
				Nup_list = list(range(_Np+1))
			
			

		if Nup_list is None:
			self._Np = None			
		else:
			self._Np = sum(Nup_list)

		self._blocks = blocks

		if blocks.get("a") is None: # by default a = 1
			blocks["a"] = 1

		if blocks.get("check_z_symm") is None or blocks.get("check_z_symm") is True:
			check_z_symm = True
		else:
			check_z_symm = False

		pblock = blocks.get("pblock")
		zblock = blocks.get("zblock")
		zAblock = blocks.get("zAblock")
		zBblock = blocks.get("zBblock")
		kblock = blocks.get("kblock")
		pzblock = blocks.get("pzblock")
		a = blocks.get("a")

		if (type(pblock) is int) and (type(zblock) is int):
			blocks["pzblock"] = pblock*zblock
			self._blocks["pzblock"] = pblock*zblock
			pzblock = pblock*zblock

		if (type(zAblock) is int) and (type(zBblock) is int):
			blocks["zblock"] = zAblock*zBblock
			self._blocks["zblock"] = zAblock*zBblock
			zblock = zAblock*zBblock



		if check_z_symm:
			# checking if spin inversion is compatible with Np and L
			if (Nup_list is not None) and ((type(zblock) is int) or (type(pzblock) is int)):
				if len(Nup_list) > 1:
					ValueError("spin inversion/particle-hole symmetry only reduces the 0 magnetization or half filled particle sector")

				Nup = Nup_list[0]

				if (L*(self.sps-1) % 2) != 0:
					raise ValueError("spin inversion/particle-hole symmetry with particle/magnetization conservation must be used with chains with 0 magnetization sector or at half filling")
				if Nup != L*(self.sps-1)//2:
					raise ValueError("spin inversion/particle-hole symmetry only reduces the 0 magnetization or half filled particle sector")

			if (Nup_list is not None) and ((type(zAblock) is int) or (type(zBblock) is int)):
				raise ValueError("zA/cA and zB/cB symmetries incompatible with magnetisation/particle symmetry")

			# checking if ZA/ZB spin inversion is compatible with unit cell of translation symemtry
			if (type(kblock) is int) and ((type(zAblock) is int) or (type(zBblock) is int)):
				if a%2 != 0: # T and ZA (ZB) symemtries do NOT commute
					raise ValueError("unit cell size 'a' must be even")

		if self._sps <= 2:
			self._pauli = pauli
			Imax = (1<<L)-1
			stag_A = sum(1<<i for i in range(0,L,2))
			#print("stag_A=", stag_A)
			stag_B = sum(1<<i for i in range(1,L,2))
			#print("stag_B=", stag_B)
			pars = _np.array([0,L,Imax,stag_A,stag_B])
			#print("pars = ", pars)
			self._operators = ("availible operators for spin_basis_1d:"+
								"\n\tI: identity "+
								"\n\t+: raising operator"+
								"\n\t-: lowering operator"+
								"\n\tx: x pauli/spin operator"+
								"\n\ty: y pauli/spin operator"+
								"\n\tz: z pauli/spin operator")

			self._allowed_ops = set(["I","+","-","x","y","z"])
			basis_1d.__init__(self,hcp_basis,hcp_ops,L,Np=Nup_list,pars=pars,count_particles=count_particles,**blocks)
		else:  # S> 1/2
			self._pauli = False
			pars = (L,) + tuple(self._sps**i for i in range(L+1)) + (1,) # flag to turn off higher spin matrix elements for +/- operators
			#print("pars = ", pars)
			self._operators = ("availible operators for spin_basis_1d:"+
								"\n\tI: identity "+
								"\n\t+: raising operator"+
								"\n\t-: lowering operator"+
								"\n\tz: z pauli/spin operator")

			self._allowed_ops = set(["I","+","-","z"])
			basis_1d.__init__(self,boson_basis,boson_ops,L,Np=Nup_list,pars=pars,count_particles=count_particles,**blocks)


	def _Op(self,opstr,indx,J,dtype):
		ME,row,col = basis_1d._Op(self,opstr,indx,J,dtype)
		if self._pauli:
			n_ops = len(opstr.replace("I",""))
			ME *= (1<<n_ops)

		return ME,row,col


	def __type__(self):
		return "<type 'qspin.basis.spin_basis_1d'>"

	def __repr__(self):
		return "< instance of 'qspin.basis.spin_basis_1d' with {0} states >".format(self._Ns)

	def __name__(self):
		return "<type 'qspin.basis.spin_basis_1d'>"


	# functions called in base class:


	def _sort_opstr(self,op):
		if op[0].count("|") > 0:
			raise ValueError("'|' character found in op: {0},{1}".format(op[0],op[1]))
		if len(op[0]) != len(op[1]):
			raise ValueError("number of operators in opstr: {0} not equal to length of indx {1}".format(op[0],op[1]))

		op = list(op)
		zipstr = list(zip(op[0],op[1]))
		if zipstr:
			zipstr.sort(key = lambda x:x[1])
			op1,op2 = zip(*zipstr)
			op[0] = "".join(op1)
			op[1] = tuple(op2)
		return tuple(op)

	def _non_zero(self,op):
		opstr = _np.array(list(op[0]))
		indx = _np.array(op[1])
		if _np.any(indx>=0):
			indx_p = indx[opstr == "+"].tolist()
			p = not any(indx_p.count(x) > 1 for x in indx_p)
			indx_p = indx[opstr == "-"].tolist()
			m = not any(indx_p.count(x) > 1 for x in indx_p)
			return (p and m)
		else:
			return True
		
	def _hc_opstr(self,op):
		op = list(op)
		# take h.c. + <--> - , reverse operator order , and conjugate coupling
		op[0] = list(op[0].replace("+","%").replace("-","+").replace("%","-"))
		op[0].reverse()
		op[0] = "".join(op[0])
		op[1] = list(op[1])
		op[1].reverse()
		op[1] = tuple(op[1])
		op[2] = op[2].conjugate()
		return self._sort_opstr(op) # return the sorted op.

	def _expand_opstr(self,op,num):
		opstr = str(op[0])
		indx = list(op[1])
		J = op[2]
 
		if len(opstr) <= 1:
			if opstr == "x":
				op1 = list(op)
				op1[0] = op1[0].replace("x","+")
				op1[2] *= 0.5
				op1.append(num)

				op2 = list(op)
				op2[0] = op2[0].replace("x","-")
				op2[2] *= 0.5
				op2.append(num)

				return (tuple(op1),tuple(op2))
			elif opstr == "y":
				op1 = list(op)
				op1[0] = op1[0].replace("y","+")
				op1[2] *= -0.5j
				op1.append(num)

				op2 = list(op)
				op2[0] = op2[0].replace("y","-")
				op2[2] *= 0.5j
				op2.append(num)

				return (tuple(op1),tuple(op2))
			else:
				op = list(op)
				op.append(num)
				return [tuple(op)]	
		else:
	 
			i = len(opstr)//2
			op1 = list(op)
			op1[0] = opstr[:i]
			op1[1] = tuple(indx[:i])
			op1[2] = complex(J)
			op1 = tuple(op1)

			op2 = list(op)
			op2[0] = opstr[i:]
			op2[1] = tuple(indx[i:])
			op2[2] = complex(1)
			op2 = tuple(op2)

			l1 = self._expand_opstr(op1,num)
			l2 = self._expand_opstr(op2,num)

			l = []
			for op1 in l1:
				for op2 in l2:
					op = list(op1)
					op[0] += op2[0]
					op[1] += op2[1]
					op[2] *= op2[2]
					l.append(tuple(op))

			return tuple(l)




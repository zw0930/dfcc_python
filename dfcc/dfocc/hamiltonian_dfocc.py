import psi4
import numpy as np

class Hamiltonian(object):
    def __init__(self, ref, aux_bas, no, nv):
        C = ref.Ca_subset("AO", "ACTIVE")
        npC = np.asarray(C)

        # Generate MO Fock matrix
        self.F = np.asarray(ref.Fa())
        self.F = npC.T @ self.F @ npC

        # Build DF tensors
        df = psi4.core.DFTensor(ref.basisset(), aux_bas, C, no, nv)
        # Transformed MO DF tensors in NumPy arrays
        self.BovQ = np.asarray(df.Qov()).swapaxes(0,1).swapaxes(1,2)
        self.BooQ = np.asarray(df.Qoo()).swapaxes(0, 1).swapaxes(1, 2)
        self.BvvQ = np.asarray(df.Qvv()).swapaxes(0, 1).swapaxes(1, 2)
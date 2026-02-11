"""
hamiltonian_fno.py: generate molecular Hamiltonian
"""
import psi4
import numpy as np

class Hamiltonian(object):
    """
    A molecular Hamiltonian object.

    Attributes
    ----------
    ref: Psi4 scf Wavefunction object
        the reference wave function built by Psi4 energy() method
    npC: NumPy array
        MO coefficient matrix for active molecular orbitals
    npC_full: NumPy array
        MO coefficient matrix for all molecular orbitals
    Fao: NumPy array
        Fock matrix in AO basis
    F: NumPy array
        Fock matrix in MO basis
    h: NumPy array
        core Hamiltonian in AO basis
    BooQ: NumPy array
        Occupied-occupied block of the molecular DF integral
    BovQ: NumPy array
        Occupied-virtual block of the molecular DF integral
    BvvQ: NumPy array
        Virtual-virtual block of the molecular DF integral
    Qso_SCF: NumPy array
        DF integral in AO basis
    Qso_CC: NumPy array
        DF integral in AO basis       
    """

    def __init__(self, ref, model, aux_bas_scf, aux_bas_cc, no, nv):
        # Save MO coefficient matrix as NumPy arrays
        C = ref.Ca_subset("AO", "ACTIVE")
        C_full = ref.Ca_subset("AO", "ALL") # save for T1-transformation
        self.npC = np.asarray(C)
        self.npC_full = np.asarray(C_full)

        # Generate MO Fock matrix
        self.Fao = np.asarray(ref.Fa()) # save F in AO basis for T1-transformation
        self.F = self.npC.T @ self.Fao @ self.npC # use in Dijab for initial t2

        # Generate core Hamiltonian
        self.mints = psi4.core.MintsHelper(ref.basisset())
        self.h = np.asarray(self.mints.ao_kinetic()) + np.asarray(
            self.mints.ao_potential())  # saves h in AO basis for T1-transformation

        # Build DF tensors
        df_scf = psi4.core.DFTensor(ref.basisset(), aux_bas_scf, C, no, nv) # save for T1-transformation 
        df_cc = psi4.core.DFTensor(ref.basisset(), aux_bas_cc, C, no, nv) # save for CC residuals

        # Transformed MO DF tensors in NumPy arrays (active occ) for initial t2
        # If doing T1-transformation in CCSD, only BovQ is needed for Dijab in initial t2
        self.BovQ = np.asarray(df_cc.Qov()).swapaxes(0,1).swapaxes(1,2)
        # Save BooQ and BvvQ for (T)
        if model == 'DF-CCSD(T)':
            self.BooQ = np.asarray(df_cc.Qoo()).swapaxes(0,1).swapaxes(1,2)
            self.BvvQ = np.asarray(df_cc.Qvv()).swapaxes(0,1).swapaxes(1,2)

        # 3-index integrals in AO basis used in T1-transformation
        self.Qso_SCF = np.asarray(df_scf.Qso())
        self.Qso_CC = np.asarray(df_cc.Qso())

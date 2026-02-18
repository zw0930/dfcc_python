"""
ccwfn_fno_ein.py: CCSD(T) T-amplitude solver (Einsums)
"""
import numpy as np
from .utils import helper_diis, fnocc, einsums_contract
import time
from .hamiltonian_fno import Hamiltonian
import itertools

class ccwfn(object):
    """
    An RHF-DF-CC wave function and energy object.

    Attributes
    ----------
    ref : Psi4 SCF Wavefunction object
        the reference wave function built by Psi4 energy() method
    eref : float
        the energy of the reference wave function (including nuclear repulsion contribution)
    model: string
        the selected CC model 
    ndocc: int
        the number of doubly occupied orbitals
    nfzc : int
        the number of frozen core orbitals
    no : int
        the number of active occupied orbitals
    nv : int
        the number of active virtual orbitals
    nact: int
        the number of active molecular orbitals
    nmo : int
        the number of active orbitals
    aux_basis_scf : Psi4 BasisSet
        the auxiliary basis for T1-transformation
    aux_basis_cc: Psi4 BasisSet
        the auxiliary basis for CC residuals
    naux_scf : int
        the number of auxiliary functions in aux_basis_scf
    naux_cc: int
        the number of auxiliary functions in aux_basis_cc
    H : Hamiltonian object
        the normal-ordered Hamiltonian, which includes the Fock matrix and DF tensors
    o : NumPy slice
        active occupied orbital subspace
    v : NumPy slice
        active virtual orbital subspace
    o_full: NumPy slice
        occupied orbital subspace
    v_full: NumPy slice
        virtual orbital subspace
    Dia : NumPy array
        one-electron energy denominator
    Dijab : NumPy array
        two-electron energy denominator
    Dijkabc: NumPy array
        three-electron energy denominator
    t1 : NumPy array
        T1 amplitudes
    t2 : NumPy array
        T2 amplitudes
    ecc | float
        the final CC correlation energy

    Methods
    -------
    solve_cc()
        Solves the CC T amplitude equations
    residuals()
        Computes the T1 and T2 residuals for a given set of amplitudes and Fock operator
    build_BooQ_LR()
        Computes the occupied-occupied block of the T1-transformed DF integral 
    build_BovQ_LR()
        Computes the occupied-virtual block of the T1-transformed DF integral
    build_BvoQ_LR()
        Computes the virtual-occupied block of the T1-transformed DF integral
    build_BvvQ_LR()
        Computes the virtual-virtual block of the T1-tranfsormed DF integral
    build_Fock_LR()
        Computes the T1-transformed Fock matrix
    cc_energy()
        Computes the CC correlation energy
    triples_correction()
        Computes the (T) correction
    """

    def __init__(self, scf_wfn, aux_bas_scf, aux_bas_cc, **kwargs):
        time_init = time.time()

        # Available CC models
        valid_cc_models = ['DF-CCSD', 'DF-CCSD(T)']
        model = kwargs.pop('model', 'DF-CCSD').upper()
        if model not in valid_cc_models:
            raise Exception("%s is not an allowed CC model." % (model))
        self.model = model

        # FNO
        self.nat_orbs = kwargs.pop('nat_orbs', False)
        # cutoff for natural virtual orbital
        self.occ_tolerance = kwargs.pop('occ_tolerance', 1E-6) # By the values of occupation number, default: 1E-6
        self.occ_percentage = kwargs.pop('occ_percentage', None) # By percentage (overrides tolerance), default: 99.0
        # By choosing number of natural orbitals (overrides tolerance and percentage), no default
        self.active_nat_orbs = kwargs.pop('active_nat_orbs', None)

        self.ref = scf_wfn
        self.eref = self.ref.energy()
        self.nfzc = self.ref.frzcpi()[0]  # assumes symmetry c1
        self.ndocc = self.ref.doccpi()[0] # all occ
        self.no = self.ndocc - self.nfzc  # active occ; assumes closed-shell
        self.nmo = self.ref.nmo()  # all MOs
        self.nv = self.nmo - self.ndocc   # virt
        self.nact = self.no + self.nv  # all active MOs
        self.aux_bas_scf = aux_bas_scf # default for SCF: "JKFIT"
        self.aux_bas_cc = aux_bas_cc # default for CC: "RI"

        self.naux_scf = aux_bas_scf.nbf()
        self.naux_cc = aux_bas_cc.nbf()

        print("NMO = %d; NACT = %d; NO = %d; NV = %d; NAUX(SCF/JK) = %d; NAUX(CC/RI) = %d" % (self.nmo, self.nact, self.no, self.nv, self.naux_scf, self.naux_cc))

        # orbital subspaces
        self.o = slice(0, self.no)
        self.v = slice(self.no, self.nact)

        # Full SCF slices (used for T1-transformation)
        self.o_full = slice(0, self.ndocc)  # SCF occ (includes frozen core)
        self.v_full = slice(self.ndocc, self.nmo)  # SCF virt

        # TODO: CHECK CONSISTENCY IF FREEZE VIRTUAL ORBITALS

        self.H = Hamiltonian(self.ref, self.model, self.aux_bas_scf, self.aux_bas_cc, self.no, self.nv)
        # Saved tensors for T1-transformation
        self.h = self.H.h # core Hamiltonian in AO basis, h_{\mu \nu}
        self.Qso_SCF = self.H.Qso_SCF # 3-index integrals in AO basis, for T1-dressed F
        self.Qso_CC = self.H.Qso_CC   # For T1-dressed B in CC residuals
        self.C_full = self.H.npC_full # MO coefficient matrix including the frozen orbital, C_{\mu \nu}

        # For computing denominators (Dij and Dijab need to be updated in each iteration with T1-dressed Fij, Fab)
        self.eps_occ = np.diag(self.H.F)[self.o]
        self.eps_virt = np.diag(self.H.F)[self.v]
        # For initial t2
        self.Dijab = self.eps_occ.reshape(-1, 1, 1, 1) + self.eps_occ.reshape(-1, 1, 1) - self.eps_virt.reshape(-1, 1) - self.eps_virt
       
        self.contract_ec = einsums_contract() # use Einsums

        if self.nat_orbs == True:
            if self.active_nat_orbs is not None:
                nv_fno = self.active_nat_orbs
            elif self.occ_percentage is not None:
                nv_fno = round(self.occ_percentage * self.nv)
            elif self.occ_tolerance is not None:
                fno = fnocc(self.Dijab, self.occ_tolerance)
                gamma_ab = fno.opdm_vv(self.contract_ec, self.H.BovQ) # build opdm virt-virt block
                nv_fno = fno.natural_orbs(gamma_ab)

            # save full virtual orbitals
            self.nv_mo = self.nv
            self.nact_mo = self.nact
            self.v_mo = self.v
            self.Dijab_mo = self.Dijab
            # initial amplitudes in MO for E_MP2^MO
            self.t1_mo = np.zeros((self.no, self.nv))
            ERIoovv = self.contract_ec('iaQ,jbQ->iajb', self.H.BovQ, self.H.BovQ).swapaxes(1, 2)
            self.t2_mo = ERIoovv / self.Dijab

            # update virtual orbitals
            self.nv = nv_fno
            self.nact = self.no + self.nv
            self.v = slice(self.no, self.nact)
            eps_virt = np.diag(self.H.F)[self.v]
            self.Dijab = eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1, 1) - eps_virt.reshape(-1, 1) - eps_virt
            # Initial amplitudes
            self.t1 = np.zeros((self.no, self.nv))
            # Form ERI[o,o,v,v]
            ERIoovv = self.contract_ec('iaQ,jbQ->iajb', self.H.BovQ[:,:nv_fno,:], self.H.BovQ[:,:nv_fno,:]).swapaxes(1, 2)
            print(self.H.BovQ[:,:nv_fno,:].shape)
            self.t2 = ERIoovv / self.Dijab
        else:
            # Initial amplitudes
            self.t1 = np.zeros((self.no, self.nv))
            ERIoovv = self.contract_ec('iaQ,jbQ->iajb', self.H.BovQ, self.H.BovQ).swapaxes(1,2)
            self.t2 = ERIoovv / self.Dijab

        print("CCWFN object initialized in %.3f seconds." % (time.time() - time_init))

    def solve_cc(self, e_conv=1e-7, r_conv=1e-7, maxiter=100, max_diis=8, start_diis=1):
        """
        Parameters
        ----------
        e_conv : float
            convergence condition for correlation energy (default if 1e-7)
        r_conv : float
            convergence condition for wave function rmsd (default if 1e-7)
        maxiter : int
            maximum allowed number of iterations of the CC equations (default is 100)
        max_diis : int
            maximum number of error vectors in the DIIS extrapolation (default is 8; set to 0 to deactivate)
        start_diis : int
            earliest iteration to start DIIS extrapolations (default is 1)

        Returns
        -------
        ecc : float
            CC correlation energy
        """
        contract = self.contract_ec

        ccsd_tstart = time.time()

        if self.nat_orbs == True:
            # MP2 energy with molecular orbitals
            ecc_mo = self.cc_energy(self.H.BovQ, self.t1_mo, self.t2_mo) # TODO: check energy expression, checked
            # MP2 energy with natural virtual orbitals
            BovQ = self.H.BovQ[:,:self.nv]
            ecc = self.cc_energy(BovQ, self.t1, self.t2)
            # second-order correction
            delta_ecc = ecc_mo - ecc
        else:
            BovQ = self.H.BovQ
            ecc = self.cc_energy(BovQ, self.t1, self.t2)

        print("CC Iter %3d: CC Ecorr = %.15f  dE = % .5E  MP2" % (0, ecc, -ecc))

        diis = helper_diis(max_diis)

        for niter in range(1, maxiter + 1):

            ecc_last = ecc

            r1, r2, Fij, Fab, BovQ = self.residuals(self.t1, self.t2)

            # Update denominators
            Dia, Dijab = self.build_denoms_from_dressed_F(Fij, Fab)
            
            t1_new = self.t1 + r1 / Dia
            t2_new = self.t2 + r2 / Dijab
            dt1 = t1_new - self.t1
            dt2 = t2_new - self.t2

            # Build DIIS error vector
            diis.add_error_vector(t1_new, t2_new, dt1, dt2)

            # DIIS extrapolation (optional)
            if max_diis > 0 and niter >= start_diis:
                t1_new, t2_new = diis.extrapolate()

            # Compute RMS from step size ||Tnew - Told||
            rms = contract('ia,ia->', dt1, dt1)
            rms += contract('ijab,ijab->', dt2, dt2)
            rms = np.sqrt(rms)

            # Update amplitudes
            self.t1 = t1_new
            self.t2 = t2_new
            
            # Recompute energy with *updated* amplitudes
            ecc = self.cc_energy(BovQ, self.t1, self.t2)
            ediff = ecc - ecc_last
            print("CC Iter %3d: CC Ecorr = %.15f  dE = % .5E  rms = % .5E"
                  % (niter, ecc, ediff, rms))

            # Convergence check
            if (abs(ediff) < e_conv) and (abs(rms) < r_conv):
                print("\nCCWFN converged in %.3f seconds.\n" % (time.time() - ccsd_tstart))
                print("E(REF)  = %20.15f" % self.eref)
                if self.nat_orbs == True:
                    ecc += delta_ecc
                if self.model == 'DF-CCSD(T)':
                    print("E(CCSD) = %20.15f" % ecc)
                    # Compute the three-electron denominator
                    Dijkabc = self.eps_occ.reshape(-1, 1, 1, 1, 1, 1) + self.eps_occ.reshape(-1, 1, 1, 1, 1) + self.eps_occ.reshape(-1, 1, 1, 1) \
                    - self.eps_virt.reshape(-1, 1, 1) - self.eps_virt.reshape(-1, 1) - self.eps_virt
                    et = self.triples_correction_2(self.o, self.v, self.H.F, self.H.BovQ, self.H.BooQ, self.H.BvvQ, self.t1, self.t2, Dijkabc)
                    print("E(T)    = %20.15f" % et)
                    ecc = ecc + et
                else:
                    print("E(%s) = %20.15f" % (self.model, ecc))
                self.ecc = ecc
                print("E(TOT)  = %20.15f" % (ecc + self.eref))
                return ecc

    def residuals(self, t1, t2):
        """
        Parameters
        ----------
        t1: NumPy array
            Current T1 amplitudes
        t2: NumPy array
            Current T2 amplitudes

        Returns
        -------
        r1, r2: NumPy arrays
            New T1 and T2 residuals: r_mu = <mu|HBAR|0>
        Fij, Fab, BovQ: NumPy arrays
            Parts of T1-transformed Hamiltonian
        """
        h = self.h
        Qso_SCF = self.Qso_SCF
        Qso_CC = self.Qso_CC
        C_full = self.C_full
        nfzc = self.nfzc
        ndocc = self.ndocc
        nmo = self.nmo
        nvirt_full = nmo - ndocc

        # Build a matrix of current t1 amplitudes (Eq. 18)
        # t1 is stored as (o_act, v_act) = (ndocc-nfzc, nv_act)
        # For LR formulas we need (v_act, o_act) 
        t1_vo_act = t1.T  # (nv_act, no_act)
        # Embed into full-space t1_full (vir_full, occ_full)
        # columns: full occupied [0..ndocc-1], active occupied are [nfzc..ndocc-1]
        t1_full = np.zeros((nvirt_full, ndocc), dtype=t1.dtype)
        # If no frozen virtuals, nv_act == nvirt_full.
        t1_full[:t1_vo_act.shape[0], nfzc:ndocc] = t1_vo_act

        # Split full MO coefficient matrix into occ/vir blocks
        Cocc = C_full[:, :ndocc]  # (nbf, ndocc)
        Cvir = C_full[:, ndocc:]  # (nbf, nvirt_full)

        # Right transform: C_R = C (1 + t) (Eq. 30)
        # occupied columns get + Cvir @ t1
        C_R_occ = Cocc + Cvir @ t1_full
        C_R_virt = Cvir

        # Left transform: C_L = C (1 - t^T) (Eq. 29)
        # virtual columns get - Cocc @ t1^T
        C_L_occ = Cocc
        C_L_virt = Cvir - Cocc @ t1_full.T

        # Assemble full C_L/C_R (AO x MO)
        C_L = np.hstack((C_L_occ, C_L_virt))
        C_R = np.hstack((C_R_occ, C_R_virt))

        occ_act = slice(nfzc, ndocc)  # length o = ndoccact
        vir_full = slice(ndocc, nmo)  # length v = nvirt_full
        C_L_act = np.hstack([C_L[:, occ_act], C_L[:, vir_full]])  # (nbf, o+v)
        C_R_act = np.hstack([C_R[:, occ_act], C_R[:, vir_full]])  # (nbf, o+v)

        # T1-dressed B integrals in full space (for T1-dressed F)
        BooQ_full_SCF = self.build_BooQ_LR(Qso_SCF, C_L_occ, C_R_occ)
        BovQ_full_SCF = self.build_BovQ_LR(Qso_SCF, C_L_occ, C_R_virt)
        BvvQ_full_SCF = self.build_BvvQ_LR(Qso_SCF, C_L_virt, C_R_virt)
        BvoQ_full_SCF = self.build_BvoQ_LR(Qso_SCF, C_L_virt, C_R_occ)

        # T1-dressed Fock matrix (in active space)
        Fij, Fia, Fai, Fab = self.build_Fock_LR(h, C_L, C_R, C_L_act, C_R_act, BooQ_full_SCF, BovQ_full_SCF, BvvQ_full_SCF, BvoQ_full_SCF)
        
        # Update F with natural virtual orbitals
        if self.nat_orbs == True:
            Fia = Fia[:, :self.nv]
            Fai = Fai[:self.nv, :]
            Fab = Fab[:self.nv, :self.nv]

        # T1-transformed B integrals in active space (for residuals)
        BooQ_full = self.build_BooQ_LR(Qso_CC, C_L_occ, C_R_occ)
        BovQ_full = self.build_BovQ_LR(Qso_CC, C_L_occ, C_R_virt)
        BvvQ_full = self.build_BvvQ_LR(Qso_CC, C_L_virt, C_R_virt)
        BvoQ_full = self.build_BvoQ_LR(Qso_CC, C_L_virt, C_R_occ) # Note that T1-transormed BvoQ does not equal to BovQ.T
        occ_act = slice(self.nfzc, self.ndocc)
        BooQ = BooQ_full[occ_act, occ_act, :]
        # update B with natural vurtial orbitals
        if self.nat_orbs == True:
            BovQ = BovQ_full[occ_act, :self.nv]
            BvoQ = BvoQ_full[:self.nv, occ_act]
            BvvQ = BvvQ_full[:self.nv, :self.nv]
        else:
            BovQ = BovQ_full[occ_act, :, :]
            BvoQ = BvoQ_full[:, occ_act, :]
            BvvQ = BvvQ_full
        
        # Permutation of t2 amplitudes, Eq. 18
        u_ijab = self.build_u_ijab(t2)
        # Permutation of 4index two-electron integrals, Eq. 17
        L_iajb = self.build_L_iajb(BovQ)
        L_aijb = self.build_L_aijb(BovQ, BvoQ, BooQ, BvvQ)

        # Build r1
        Aia = self.build_Aia(u_ijab, BovQ, BvvQ) # Eq. 20
        Bia = self.build_Bia(u_ijab, BooQ, BovQ) # Eq. 21
        Cia = self.build_Cia(Fia, u_ijab) # Eq. 22
        r1 = self.r_T1(Fai, Aia, Bia, Cia) # Eq. 19

        # Build r2
        A_ijab = self.A_ijab(t2, BvvQ) # Eq. 11
        B_ijab = self.B_ijab(t2, BooQ, BovQ) # Eq. 12
        C_ijab = self.C_ijab(t2, BooQ, BovQ, BvvQ) # Eq. 13
        D_ijab = self.D_ijab(L_iajb, L_aijb, u_ijab) # Eq. 14
        E_ijab = self.E_ijab(t2, u_ijab, Fab, BovQ) # Eq. 15
        G_ijab = self.G_ijab(t2, u_ijab, Fij, BovQ) # Eq. 16
        r2 = self.r_T2(BvoQ, A_ijab, B_ijab, C_ijab, D_ijab, E_ijab, G_ijab) # Eq.10

        return r1, r2, Fij, Fab, BovQ

    # T1-transformation
    # Eq. 25-27
    def build_BooQ_LR(self, Qso, C_L_occ, C_R_occ):
        """
        Build T1-transformed occupied-occupied 3-index DF integral from AO 3-index integrals.

        Parameters
        ----------
        Qso : NumPy array
            AO 3-index tensor, shape (nQ, nbf, nbf),
            i.e. Qso[Q, mu, nu] = (Q|mu nu).
        C_L_occ : NumPy array
            Left MO coefficients (occupied block), shape (nbf, nocc).
        C_R_occ : NumPy array
            Right MO coefficients (occupied block), shape (nbf, nocc).

        Returns
        -------
        BooQ : NumPy array
            LR MO 3-index tensor for oo block, shape (nocc, nocc, nQ),
            i.e. BooQ[i, j, Q] = sum_{mu,nu} C_L_occ[mu,i] * Qso[Q,mu,nu] * C_R_occ[nu,j].
        """
        Qij = np.einsum("Qmn,mi,nj->Qij", Qso, C_L_occ, C_R_occ, optimize=True)
        # return in the preferred layout (i, j, Q)
        return Qij.transpose(1, 2, 0)

    def build_BovQ_LR(self, Qso, C_L_occ, C_R_virt):
        """
        Build T1-transformed occupied-virtual 3-index DF integral from AO 3-index integrals.

        Parameters
        ----------
        Qso : NumPy array
            AO 3-index tensor, shape (nQ, nbf, nbf),
            i.e. Qso[Q, mu, nu] = (Q|mu nu).
        C_L_occ : NumPy array
            Left MO coefficients (occupied block), shape (nbf, nocc).
        C_R_virt : NumPy array
            Right MO coefficients (virtual block), shape (nbf, nvirt).

        Returns
        -------
        BovQ : NumPy array
            LR MO 3-index tensor for ov block, shape (nocc, nvirt, nQ),
            i.e. BovQ[i, a, Q] = sum_{mu,nu} C_L_occ[mu,i] * Qso[Q,mu,nu] * C_R_virt[nu,a].
        """
        Qia = np.einsum("Qmn,mi,na->Qia", Qso, C_L_occ, C_R_virt, optimize=True)
        return Qia.transpose(1, 2, 0)  # (i, a, Q)

    def build_BvvQ_LR(self, Qso, C_L_virt, C_R_virt):
        """
        Build T1-transformed virtual-virtual 3-index DF integral from AO 3-index integrals.

        Parameters
        ----------
        Qso : NumPy array
            AO 3-index tensor, shape (nQ, nbf, nbf),
            i.e. Qso[Q, mu, nu] = (Q|mu nu).
        C_L_virt : NumPy array
            Left MO coefficients (virtual block), shape (nbf, nvirt).
        C_R_virt : NumPy array
            Right MO coefficients (virtual block), shape (nbf, nvirt).

        Returns
        -------
        BvvQ : NumPy array
            LR MO 3-index tensor for ov block, shape (nvirt, nvirt, nQ),
            i.e. BvvQ[a, b, Q] = sum_{mu,nu} C_L_occ[mu,a] * Qso[Q,mu,nu] * C_R_virt[nu,b].
        """
        Qab = np.einsum("Qmn,ma,nb->Qab", Qso, C_L_virt, C_R_virt, optimize=True)
        return Qab.transpose(1, 2, 0)  # (a, b, Q)

    def build_BvoQ_LR(self, Qso, C_L_virt, C_R_occ):
        """
        Build T1-transformed virtual-occupied 3-index DF integral from AO 3-index integrals.

        Parameters
        ----------
        Qso : NumPy array
            AO 3-index tensor, shape (nQ, nbf, nbf),
            i.e. Qso[Q, mu, nu] = (Q|mu nu).
        C_L_virt : NumPy array
            Left MO coefficients (virtual block), shape (nbf, nvirt).
        C_R_occ : NumPy array
            Right MO coefficients (occupied block), shape (nbf, nocc).

        Returns
        -------
        BvvQ : NumPy array
            LR MO 3-index tensor for ov block, shape (nvirt, nocc, nQ),
            i.e. BvvQ[a, i, Q] = sum_{mu,nu} C_L_occ[mu,a] * Qso[Q,mu,nu] * C_R_occ[nu,i].
        """
        Qai = np.einsum("Qmn,ma,ni->Qai", Qso, C_L_virt, C_R_occ, optimize=True)
        return Qai.transpose(1, 2, 0)  # (a, i, Q)

    def build_Fock_LR(self, h_ao, C_L, C_R, C_L_act, C_R_act, BooQ, BovQ, BvvQ, BvoQ):
        """
        Build t1-dressed (LR) Fock blocks in active space.

        Parameters
        ----------
        h_ao : (nbf, nbf) array
            AO core Hamiltonian (T + V).
        C_L, C_R : (nbf, nmo) arrays
            LR-dressed AO->MO coefficient matrices.
        BooQ : (ndocc, ndocc, nQ)
        BovQ : (ndocc, nvirt_full, nQ)
        BvvQ : (nvirt_full, nvirt_full, nQ)
        BvoQ : (nvirt_full, ndocc, nQ)

        Returns
        -------
        Fij, Fia, Fai, Fab : active-space blocks
            Fij: (no_act, no_act)
            Fia: (no_act, nv_act)
            Fai: (nv_act, no_act)
            Fab: (nv_act, nv_act)
        """
        contract = self.contract_ec

        nfzc = self.nfzc
        ndocc = self.ndocc
        nmo = self.nmo
        nvirt_full = nmo - ndocc
        no = ndocc - nfzc
        nv = nmo - ndocc

        # First term in Eq. 25 (T1-transformed core Hamiltonian)
        # Eq. 26: h^LR = C_L^T h C_R (full MO)
        # hLR: (nmo, nmo)
        tmp = contract("pm,pq->mq", C_L_act, h_ao)
        hLR = contract("mq,qn->mn", tmp, C_R_act)

        # Split full blocks
        h_oo = hLR[:no, :no]  # (ndocc, ndocc)
        h_ov = hLR[:no, no:]  # (ndocc, nvirt_full)
        h_vo = hLR[no:, :no]  # (nvirt_full, ndocc)
        h_vv = hLR[no:, no:]  # (nvirt_full, nvirt_full)
        
        # Second term in Eq. 25
        # 2 * (sum_k B_kk^Q) * B_rs^Q  
        JQ = np.einsum("kkQ->Q", BooQ)

        J_oo = 2.0 * contract("ijQ,Q->ij", BooQ, JQ)
        J_ov = 2.0 * contract("iaQ,Q->ia", BovQ, JQ)
        J_vo = 2.0 * contract("aiQ,Q->ai", BvoQ, JQ)
        J_vv = 2.0 * contract("abQ,Q->ab", BvvQ, JQ)

        # Third term in Eq. 25
        # sum_{kQ} B_{r k}^Q B_{k s}^Q
        # K_oo[i,j] = sum_{kQ} BooQ[i,k,Q] * BooQ[j,k,Q]
        K_oo = contract("ikQ,kjQ->ij", BooQ, BooQ)
        # K_ov[i,a] = sum_{kQ} BooQ[i,k,Q] * BovQ[k,a,Q]
        K_ov = contract("ikQ,kaQ->ia", BooQ, BovQ)
        # K_vo[a,i] = sum_{kQ} BvoQ[a,k,Q] * BooQ[i,k,Q]
        K_vo = contract("akQ,kiQ->ai", BvoQ, BooQ)
        # K_vv[a,b] = sum_{kQ} BvoQ[a,k,Q] * BovQ[k,b,Q]
        K_vv = contract("akQ,kbQ->ab", BvoQ, BovQ)

        temp_oo = J_oo - K_oo
        temp_ov = J_ov - K_ov
        temp_vo = J_vo - K_vo
        temp_vv = J_vv - K_vv

        occ_act = slice(nfzc, ndocc)  # active occupied within full occupied
        # T1-transformed Fock matrix in active MO space
        Fij = h_oo + temp_oo[occ_act, occ_act]
        Fia = h_ov + temp_ov[occ_act, :]
        Fai = h_vo + temp_vo[:, occ_act]
        Fab = h_vv + temp_vv

        return Fij, Fia, Fai, Fab

    # u_pq^rs = 2 * t_pq^rs  - t_pq^sr, Eq. 18
    def build_u_ijab(self, t2):
        return 2 * t2 - t2.swapaxes(2, 3)

    # permuted TEI
    # Lpqrs = 2 * sum_Q(BpqQ BrsQ) - sum_Q(BpsQ BrqQ), Eq. 17
    def build_L_iajb(self, BovQ):
        contract = self.contract_ec
        tmp = contract('iaQ,jbQ->iajb', BovQ, BovQ)
        return 2 * tmp - tmp.swapaxes(1, 3)
    # Note that L_aijb does not equal to L_iajb
    def build_L_aijb(self, BovQ, BvoQ, BooQ, BvvQ):
        contract = self.contract_ec
        tmp1 = contract('aiQ,jbQ->aijb', BvoQ, BovQ)
        tmp2 = contract('abQ,jiQ->abji', BvvQ, BooQ).transpose(0,3,2,1)
        return 2 * tmp1 - tmp2

    def build_Tau_ijab(self, t1, t2):
        contract = self.contract_ec
        return t2 + contract('ia,jb->ijab', t1, t1)

    def build_denoms_from_dressed_F(self, Fij, Fab):
        eps_occ = np.diag(Fij)  # (no,)
        eps_vir = np.diag(Fab)  # (nv,)

        Dia = eps_occ[:, None] - eps_vir[None, :]  # (no,nv)

        Dijab = (
                eps_occ[:, None, None, None] + eps_occ[None, :, None, None]
                - eps_vir[None, None, :, None] - eps_vir[None, None, None, :]
        )  # (no,no,nv,nv)

        return Dia, Dijab

    # TODO: rewrite the sym/anti-sym algorithm for certain intermediates
    def _idx2(self, i, j):
        return i * (i + 1) // 2 + j

    # Intermediates for r1, r2
    # r1
    # Aia = sum_dQ(sum_kc(u_ki^cd kcQ) BadQ)
    def build_Aia(self, u_ijab, BovQ, BvvQ):
        contract = self.contract_ec
        tmp = contract('kicd,kcQ->idQ', u_ijab, BovQ)
        return contract('idQ,adQ->ia', tmp, BvvQ)

    # Bia = - sum_klc(u_klac (sum_Q (BkiQ BlcQ))
    def build_Bia(self, u_ijab, BooQ, BovQ):
        contract = self.contract_ec
        tmp = contract('kiQ,lcQ->kilc', BooQ, BovQ)
        Bia = -1 * contract('kilc,klac->ia', tmp, u_ijab)
        return Bia

    # Cia = sum_kc (F_kc u_ikac)
    def build_Cia(self, Fia, u_ijab):
        contract = self.contract_ec
        return contract('kc,ikac->ia', Fia, u_ijab)

    # r1 = Fia + Aia + Bia + Cia
    def r_T1(self, Fai, Aia, Bia, Cia):
        r1 = Fai.copy().T
        r1 += Aia + Bia + Cia
       
        return r1

    # r2
    # A_{ij}^{ab} = \sum_{cd}(t_{ij}^{cd} (\sum_Q (B_{ac}^Q) B_{bd}^Q))
    def A_ijab(self, t2, BvvQ):
        contract = self.contract_ec
        # TODO: sym/anti-sym tensors for ladder-ladder term
        tmp = contract('acQ,bdQ->acbd', BvvQ, BvvQ)
        return contract('ijcd, acbd->ijab', t2, tmp)

    # Bijab = sum_kl (t_kl^ab [(sum_Q(BkiQ BljQ)) + (sum_cd [t_ijcd (sum_Q(BkcQ BldQ))])])
    # B_{ij}^{ab} = \sum_{kl} (t_{kl}^{ab} ())
    def B_ijab(self, t2, BooQ, BovQ):
        contract = self.contract_ec
        tmp1 = contract('kiQ, ljQ->kilj', BooQ, BooQ).transpose(1,3,0,2) # ijkl
        tmp2 = contract('kcQ, ldQ->kcld', BovQ, BovQ)
        tmp1 = contract('ijcd,kcld->ijkl', t2, tmp2, beta=1.0, out=tmp1)
        Bijab = contract('ijkl,klab->ijab', tmp1, t2)

        return Bijab

    # Cijab = - sum_kc (t_kj^bc [(sum_Q(BkiQ BacQ)) + (-1 / 2 * sum_ld [t_liad (sum_Q(BkdQ BlcQ))])])
    def C_ijab(self, t2, BooQ, BovQ, BvvQ):
        contract = self.contract_ec
        V1 = contract("kiQ,acQ->kiac", BooQ, BvvQ)
        V2 = contract("kdQ,lcQ->kdlc", BovQ, BovQ)
        tmp = contract("liad,kdlc->iakc", t2, V2).transpose(2,0,1,3)
        V1 -= 0.5 * tmp
        Cijab = contract("kiac,kjbc->iajb", V1, t2, alpha=-1.0).swapaxes(1,2)

        return Cijab

    # Dijab = - 1 / 2 * sum_kc [u_jkbc (Laikc + 1 / 2 * (sum_ld[u_ilad L_ldkc]))]
    def D_ijab(self, L_iajb, L_aijb, u_ijab):
        contract = self.contract_ec
        tmp = L_aijb.swapaxes(0,1).copy()  # L_aikc ->iakc
        tmp = contract('ilad,ldkc->iakc', u_ijab, L_iajb, alpha=0.5, beta=1.0, out=tmp)
        Dijab = contract('jkbc,iakc->jbia', u_ijab, tmp, alpha=0.5).transpose(2,0,3,1)

        return Dijab

    # Eijab = sum_c (t_ijac [Fbc - sum_kld(u_klbd [sum_Q (BldQ BkcQ)])])
    def E_ijab(self, t2, u_ijab, Fae, BovQ):
        contract = self.contract_ec
        tmp = Fae.copy()
        tmp1 = contract('ldQ,kcQ->ldkc', BovQ, BovQ)
        tmp = contract('klbd,ldkc->bc', u_ijab, tmp1, alpha=-1.0, beta=1.0, out=tmp)
        
        return contract('ijac, bc->ijab', t2, tmp)

    # Gijab = - sum_k (t_ikab [Fkj - sum_lcd(u_ljcd [sum_Q (BkdQ BlcQ)])])
    # TODO: typo in Gijab = - sum_k (t_ikab [Fkj + (not -) sum_lcd(u_ljcd [sum_Q (BkdQ BlcQ)])])
    def G_ijab(self, t2, u_ijab, Fij, BovQ):
        contract = self.contract_ec
        tmp = Fij.copy()
        tmp1 = contract('kdQ,lcQ->kdlc', BovQ, BovQ)
        tmp = contract('kdlc,ljcd->kj', tmp1, u_ijab, beta=1.0, out=tmp)
        Gijab = contract('ikab,kj->iabj', t2, tmp, alpha=-1.0).transpose(0,3,1,2)

        return Gijab

    # r2 = sum_Q (BiaQ BjbQ) + A_ijab + B_ijab + Pijab (1 / 2 * C_ijab + C_jiab + D_ijab + E_ijab + G_ijab)
    def r_T2(self, BvoQ, A_ijab, B_ijab, C_ijab, D_ijab, E_ijab, G_ijab):
        contract = self.contract_ec
        r2 = contract('aiQ, bjQ->aibj', BvoQ, BvoQ).transpose(1,3,0,2)
        tmp = 0.5 * C_ijab.copy()
        tmp += C_ijab.swapaxes(0, 1) + D_ijab + E_ijab + G_ijab
        r2 += A_ijab + B_ijab + tmp + tmp.swapaxes(0, 1).swapaxes(2, 3)

        return r2

    def cc_energy(self, BovQ, t1, t2):
        contract = self.contract_ec
        Lijab = self.build_L_iajb(BovQ).swapaxes(1,2)
        tau = self.build_Tau_ijab(t1, t2)
        return contract('ijab,ijab->', Lijab, tau)

    # Permutation operator P_ijk^abc
    def P_ijkabc(self, X):
        assert X.ndim == 6, "X must be a rank-6 tensor (i,j,k,a,b,c)."

        result = np.zeros_like(X)

            # Permute the first 3 indices (i,j,k) and apply the same permutation
            # to the last 3 indices (a,b,c)
        for p in itertools.permutations(range(3)):  # p is a tuple like (0,1,2), (1,2,0), ...
            order = p + tuple(i + 3 for i in p)  # e.g. (0,1,2,3,4,5) or (1,2,0,4,5,3)
            result += X.transpose(order)

        return result
    
    # (T) correction
    def triples_correction_1(self, o, v, F, BovQ, BooQ, BvvQ, t1, t2, Dijkabc):
        contract  = self.contract_ec
        # Calculate 4-index integrals
        Vovvv = contract('iaQ,ebQ->iaeb', BovQ, BvvQ)
        Vooov = contract('mjQ,kcQ->mjkc', BooQ, BovQ)
        Vovov = contract('jbQ,kcQ->jbkc', BovQ, BovQ)
        # Build W
        W = contract('iaeb,jkec->iabjkc', Vovvv, t2)
        W = contract('imab,mjkc->iabjkc', t2, Vooov, alpha=-1.0, beta=1.0, out=W).transpose(0, 3, 4, 1, 2, 5)
        Wijkabc = self.P_ijkabc(W)

        #Wijkabc = W.copy()
        # Build V
        Fme = F[o,v].copy()
        Vijkabc = Wijkabc.copy()
        Vijkabc += contract('ia,jbkc->iajbkc', t1, Vovov).transpose(0, 2, 4, 1, 3, 5)
        Vijkabc += contract('jb,iakc->jbiakc', t1, Vovov).transpose(2, 0, 4, 3, 1, 5)
        Vijkabc += contract('kc,iajb->kciajb', t1, Vovov).transpose(2, 4, 0, 3, 5, 1)        
        #Vijkabc += contract('ia,jkbc->ijkabc', Fme, t2)
        #Vijkabc += contract('jb,ikac->ijkabc', Fme, t2)
        #Vijkabc += contract('kc,ijab->ijkabc', Fme, t2)

        # 4 * W_ijk^abc + W_ijk^bca + W_ijk^cab
        W_tmp = 4 * Wijkabc + Wijkabc.swapaxes(3,4).swapaxes(4,5) + Wijkabc.swapaxes(4,5).swapaxes(3,4)
        # V_ijk^abc - V_ijk^cba
        V_tmp = Vijkabc - Vijkabc.swapaxes(3,5)
        # Apply denominator
        W_tmp /= Dijkabc
        print("check Dijkabc: ", Dijkabc.shape)

        et = contract('ijkabc,ijkabc->', W_tmp, V_tmp, alpha=1 / 3)

        return et

    def triples_correction_2(self, o, v, F, BovQ, BooQ, BvvQ, t1, t2, Dijkabc):
        """(T) energy using the X/Y/Z / restricted-sum algorithm (eq. 13–17).

        Parameters
        ----------
        o, v : slices
            Occupied and virtual slices into the MO dimension of F.
        F : (nmo, nmo)
            Fock matrix in MO basis.
        BovQ : (no, nv, nQ)
            DF 3-index integrals B_{ia}^Q.
        BooQ : (no, no, nQ)
            DF 3-index integrals B_{ij}^Q.
        BvvQ : (nv, nv, nQ)
            DF 3-index integrals B_{ab}^Q.
        t1 : (no, nv)
            Singles amplitudes t_i^a.
        t2 : (no, no, nv, nv)
            Doubles amplitudes t_{ij}^{ab}.
        Dijkabc : (no, no, no, nv, nv, nv)
            Triples energy denominators D_{ijk}^{abc}.

        Returns
        -------
        float
            (T) energy correction.
        """
        contract = self.contract_ec
        no, nv = t1.shape

        # --------------------------------------------------------------
        # 1. Build four-index intermediates (same as your working code)
        # --------------------------------------------------------------
        # (ov|vv) ≡ <ia|eb>
        Vovvv = contract('iaQ,ebQ->iaeb', BovQ, BvvQ)
        # (oo|ov) ≡ <mj|kc>
        Vooov = contract('mjQ,kcQ->mjkc', BooQ, BovQ)
        # (ov|ov) ≡ <jb|kc>
        Vovov = contract('jbQ,kcQ->jbkc', BovQ, BovQ)

        # --------------------------------------------------------------
        # 2. Build W_{ijk}^{abc} as in the second algorithm
        #    W_raw(ijk,abc) = sum_e <ib|ae> t_{jk}^{ec} - sum_m <jk|mc> t_{im}^{ab}
        # --------------------------------------------------------------
        W = contract('iaeb,jkec->iabjkc', Vovvv, t2)
        W = contract('imab,mjkc->iabjkc', t2, Vooov, alpha=-1.0, beta=1.0, out=W).transpose(0,3,4,1,2,5)

        # Apply permutation operator P_{ijk}^{abc} so W is symmetrized
        # in (i,j,k) and (a,b,c), matching the derivation in the paper.
        W = self.P_ijkabc(W)

        # --------------------------------------------------------------
        # 3. Build V_{ijk}^{abc} from W (again mirroring your working code)
        #    V = W + t1 * (ov|ov)  (+ optional F·t2 terms if you want them)
        # --------------------------------------------------------------
        Fme = F[o, v].copy()

        V = W.copy()
        V += contract('ia,jbkc->iajbkc', t1, Vovov).transpose(0,2,4,1,3,5)
        V += contract('jb,iakc->jbiakc', t1, Vovov).transpose(2,0,4,3,1,5)
        V += contract('kc,iajb->kciajb', t1, Vovov).transpose(2,4,0,3,5,1)

        # If you decide to include the F·t2 pieces, uncomment:
        # V += contract('ia,jkbc->ijkabc', Fme, t2)
        # V += contract('jb,ikac->ijkabc', Fme, t2)
        # V += contract('kc,ijab->ijkabc', Fme, t2)

        # (Optionally, you can also symmetrize V with P_ijkabc,
        # but for this derivation W-symmetrization is enough to match
        # the "second" algorithm. If you want to be extra safe, do:)
        # V = self.P_ijkabc(V)

        # --------------------------------------------------------------
        # 4. Build Ṽ_{ijk}^{abc} according to eq. (17):
        #    Ṽ^{abc} = V^{abc} / (1 + δ_{abc}),
        #    where δ_{abc} = δ_{ab} + δ_{bc} + δ_{ac}.
        # --------------------------------------------------------------
        delta_abc = np.zeros((nv, nv, nv), dtype=int)
        for a in range(nv):
            for b in range(nv):
                for c in range(nv):
                    delta_abc[a, b, c] = (
                            (a == b) + (b == c) + (a == c)
                    )

        V_tilde = V / (1.0 + delta_abc.reshape(1, 1, 1, nv, nv, nv))

        # --------------------------------------------------------------
        # 5. Evaluate eq. (13) with restricted sums:
        #    i ≥ j ≥ k, a ≥ b ≥ c.
        #    X, Y, Z, W^(Y), W^(Z) as in eqs. (14–16).
        # --------------------------------------------------------------
        E_T = 0.0

        for i in range(no):
            for j in range(i + 1):  # j <= i
                for k in range(j + 1):  # k <= j
                    # δ_ijk = δ_ij + δ_jk + δ_ik
                    delta_ijk = int(i == j) + int(j == k) + int(i == k)
                    pref_ijk = 2.0 - delta_ijk

                    for a in range(nv):
                        for b in range(a + 1):  # b <= a
                            for c in range(b + 1):  # c <= b

                                # Helper to fetch W and Ṽ for permutations of (a,b,c)
                                def WV(pa, pb, pc):
                                    return (
                                        W[i, j, k, pa, pb, pc],
                                        V_tilde[i, j, k, pa, pb, pc],
                                    )

                                # All six permutations of (a,b,c)
                                Wabc, Vabc = WV(a, b, c)
                                Wacb, Vacb = WV(a, c, b)
                                Wbac, Vbac = WV(b, a, c)
                                Wbca, Vbca = WV(b, c, a)
                                Wcab, Vcab = WV(c, a, b)
                                Wcba, Vcba = WV(c, b, a)

                                # X_ijk^{abc} (eq. 14)
                                X = (
                                        Wabc * Vabc
                                        + Wacb * Vacb
                                        + Wbac * Vbac
                                        + Wbca * Vbca
                                        + Wcab * Vcab
                                        + Wcba * Vcba
                                )

                                # Y_ijk^{abc} (eq. 15)
                                Y = Vabc + Vbca + Vcab

                                # Z_ijk^{abc} (eq. 16)
                                Z = Vacb + Vbac + Vcba

                                # W^(Y) and W^(Z)
                                WY = Wabc + Wbca + Wcab
                                WZ = Wacb + Wbac + Wcba

                                numer = (
                                        (Y - 2.0 * Z) * WY
                                        + (Z - 2.0 * Y) * WZ
                                        + 3.0 * X
                                )

                                E_T += (
                                        pref_ijk
                                        * numer
                                        / Dijkabc[i, j, k, a, b, c]
                                )

        return E_T

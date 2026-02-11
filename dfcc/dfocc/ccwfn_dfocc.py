"""
ccwfn.py: CCSD(T) T-amplitude solver
Reference:
"""
import numpy as np
import time
import opt_einsum as opt
from .utils import helper_diis
from .hamiltonian_dfocc import Hamiltonian
from .three_index_intr import *
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
    nfzc : int
        the number of frozen core orbitals
    no : int
        the number of active occupied orbitals
    nv : int
        the number of active virtual orbitals
    nmo : int
        the number of active orbitals
    naux : int
        the number of auxiliary functions
    aux_bas : Psi4 BasisSet
        the auxiliary basis for CC iterations "RIFIT"
    H : Hamiltonian object
        the normal-ordered Hamiltonian, which includes the Fock matrix and DF tensors
    o : NumPy slice
        occupied orbital subspace
    v : NumPy slice
        virtual orbital subspace
    Dia : NumPy array
        one-electron energy denominator
    Dijab : NumPy array
        two-electron energy denominator
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
    """

    def __init__(self, scf_wfn, aux_bas, **kwargs):
        time_init = time.time()

        # Available CC models
        valid_cc_models = ['DF-CCSD', 'DF-CCSD(T)']
        model = kwargs.pop('model', 'DF-CCSD').upper()
        if model not in valid_cc_models:
            raise Exception("%s is not an allowed CC model." % (model))
        self.model = model

        self.ref = scf_wfn
        self.eref = self.ref.energy()
        self.nfzc = self.ref.frzcpi()[0]  # assumes symmetry c1
        self.no = self.ref.doccpi()[0] - self.nfzc  # active occ; assumes closed-shell
        self.nmo = self.ref.nmo()  # all MOs
        self.nv = self.nmo - self.no - self.nfzc  # active virt
        self.nact = self.no + self.nv  # all active MOs
        self.aux_bas = aux_bas
        self.naux = aux_bas.nbf()  # auxiliary basis set for DF

        print("NMO = %d; NACT = %d; NO = %d; NV = %d; NAUX = %d" % (self.nmo, self.nact, self.no, self.nv, self.naux))

        # orbital subspaces
        self.o = slice(0, self.no)
        self.v = slice(self.no, self.nact)

        # For convenience
        o = self.o
        v = self.v

        # Get MOs
        #self.C = self.ref.Ca_subset("AO", "ACTIVE")

        self.H = Hamiltonian(self.ref, self.aux_bas, self.no, self.nv)

        # Get denominators
        eps_occ = np.diag(self.H.F)[o]
        eps_virt = np.diag(self.H.F)[v]
        self.Dia = eps_occ.reshape(-1, 1) - eps_virt
        self.Dijab = eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1, 1) - eps_virt.reshape(-1, 1) - eps_virt
        if self.model == 'DF-CCSD(T)':
            self.Dijkabc = eps_occ.reshape(-1, 1, 1, 1, 1, 1) + eps_occ.reshape(-1, 1, 1, 1, 1) + eps_occ.reshape(-1, 1, 1, 1) \
                           - eps_virt.reshape(-1, 1, 1) - eps_virt.reshape(-1, 1) - eps_virt
        self.contract = opt.contract

        # Initial amplitudes
        self.t1 = np.zeros((self.no, self.nv))
        # Form ERI[o,o,v,v]
        ERIoovv = self.contract('iaQ,jbQ->iajb', self.H.BovQ, self.H.BovQ).swapaxes(1, 2)
        self.t2 = ERIoovv / self.Dijab

        print("CCWFN object initialized in %.3f seconds." % (time.time() - time_init))

        self.Fia = self.H.F[self.o, self.v]
        self.Fij = self.H.F[self.o, self.o]
        self.Fab = self.H.F[self.v, self.v]

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
        ccsd_tstart = time.time()

        o = self.o
        v = self.v
        F = self.H.F
        BovQ = self.H.BovQ

        print("BovQ: ", BovQ[0,0,0])
        BooQ = self.H.BooQ
        BvvQ = self.H.BvvQ
        Dia = self.Dia
        Dijab = self.Dijab
        if self.model == 'DF-CCSD(T)':
            Dijkabc = self.Dijkabc

        contract = self.contract

        ecc = self.cc_energy(o, v, F, BovQ, self.t1, self.t2)
        print("CC Iter %3d: CC Ecorr = %.15f  dE = % .5E  MP2" % (0, ecc, -ecc))

        diis = helper_diis(self.t1, self.t2, max_diis)

        for niter in range(1, maxiter + 1):

            ecc_last = ecc

            r1, r2 = self.residuals(self.t1, self.t2)

            t1_new = r1 / Dia
            t2_new = r2 / Dijab

            # Build DIIS error vector
            diis.add_error_vector(t1_new, t2_new)

            # DIIS extrapolation (optional)
            if max_diis > 0 and niter >= start_diis:
                t1_new, t2_new = diis.extrapolate(t1_new, t2_new)

            # Compute RMS from step size ||Tnew - Told||
            dt1 = t1_new - self.t1
            dt2 = t2_new - self.t2
            rms = contract('ia,ia->', dt1, dt1)
            rms += contract('ijab,ijab->', dt2, dt2)
            rms = np.sqrt(rms)

            # Update amplitudes
            self.t1 = t1_new
            self.t2 = t2_new

            # Recompute energy with *updated* amplitudes
            ecc = self.cc_energy(o, v, F, BovQ, self.t1, self.t2)
            ediff = ecc - ecc_last
            print("CC Iter %3d: CC Ecorr = %.15f  dE = % .5E  rms = % .5E"
                  % (niter, ecc, ediff, rms))

            # Convergence check
            if (abs(ediff) < e_conv) and (abs(rms) < r_conv):
                print("\nCCWFN converged in %.3f seconds.\n" % (time.time() - ccsd_tstart))
                print("E(REF)  = %20.15f" % self.eref)
                if self.model == 'DF-CCSD(T)':
                    print("E(CCSD) = %20.15f" % ecc)                  
                    et = self.triples_correction_2(o, v, F, BovQ, BooQ, BvvQ, self.t1, self.t2, Dijkabc)
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
        """

        contract = self.contract

        o = self.o
        v = self.v
        no = self.no
        nv = self.nv
        F = self.H.F
        BooQ = self.H.BooQ
        BovQ = self.H.BovQ
        BvvQ = self.H.BvvQ

        u_ijab = self.build_u_ijab(t2)
        tau_ijab = self.build_tau_ijab(t1)
        Tau_ijab = self.build_Tau_ijab(t1, t2)
        Tau_tilde_ijab = self.build_Tau_tilde_ijab(t1, t2)

        # build 3-index intermediates
        tijQ = build_tijQ(t1, BovQ, contract)
        tiaQ = build_tiaQ(t1, BvvQ, contract)
        taiQ = build_taiQ(t1, BooQ, contract)
        tabQ = build_tabQ(t1, BovQ, contract)
        tQ = build_tQ(t1, BovQ, contract)
        TiaQ = build_TiaQ(u_ijab, BovQ, contract)
        TiaQ_tilde = build_TiaQ_tilde(t1, tijQ, contract)
        tiaQ_prime = build_tiaQ_prime(tiaQ, taiQ, TiaQ_tilde)
        Tau_iaQ = build_Tau_iaQ(Tau_tilde_ijab, BovQ, contract)
        Tau_iaQ_prime = build_Tau_iaQ_prime(tiaQ, Tau_iaQ)
        Tau_iaQ_dounble_prime = build_Tau_iaQ_double_prime(taiQ, Tau_iaQ)

        # build t1-dressed Fock matrix
        Fme = self.build_Fme(tQ, BovQ, tijQ)
        Fmi = self.build_Fmi(tQ, BooQ, BovQ, Tau_iaQ_dounble_prime, t1, Fme)
        Fae = self.build_Fae(tQ, BovQ, BvvQ, Tau_iaQ_prime, t1, Fme)

        # Build 4-index intermediates for T2 residual
        Xijab = self.build_Xijab(t2, Fae, Fmi, tiaQ_prime, tiaQ, taiQ, BovQ)
        # Build W intermediates
        # Wmnij
        #WmnijT2_1 = self.build_WmnijT2(BooQ, BovQ, tijQ, Tau_ijab)
        WmnijT2 = self.build_WmnijT2_packed(no, nv, BooQ, BovQ, tijQ, Tau_ijab)
        # Wmbej
        WmbejT2 = self.build_WmbejT2(BovQ, BooQ, BvvQ, tijQ, tabQ, tiaQ_prime, TiaQ, t2, u_ijab)
        # Wijam
        #WijamT2_1 = self.build_WijamT2(BvvQ, BovQ, t1, Tau_ijab)
        WijamT2 = self.build_WijamT2_packed(no, nv, BvvQ, BovQ, t1, Tau_ijab)
        # Wabef
        #WabefT2_1 = self.build_WabefT2(BvvQ,Tau_ijab)
        WabefT2 = self.build_WabefT2_packed(no, nv, BvvQ, Tau_ijab)

        r1 = self.r_T1(t1, Fmi, Fme, Fae, u_ijab, tQ, TiaQ, tiaQ, BooQ, BovQ, BvvQ)
        r2 = self.r_T2(Xijab, BovQ, WmnijT2, WmbejT2, WijamT2, WabefT2)

        return r1, r2

    def build_u_ijab(self, t2):
        return 2 * t2 - t2.swapaxes(0, 1)

    def build_tau_ijab(self, t1):
        contract = self.contract
        return 0.5 * contract('ia,jb->ijab', t1, t1)

    def build_Tau_ijab(self, t1, t2):
        contract = self.contract
        return t2 + contract('ia,jb->ijab', t1, t1)

    def build_Tau_tilde_ijab(self, t1, t2):
        contract = self.contract
        return t2 + 0.5* contract('ia,jb->ijab', t1, t1)

    def build_Fme(self, tQ, BovQ, tijQ):
        contract = self.contract
        F = contract('meQ,Q->me', BovQ, tQ)
        F -= contract('nmQ,neQ->me', tijQ, BovQ)
        return F

    def build_Fmi(self, tQ, BooQ, BovQ, Tau_iaQ_dounble_prime, t1, Fme):
        contract = self.contract
        F = contract('miQ,Q->mi', BooQ, tQ)
        F += contract('meQ,ieQ->mi', BovQ, Tau_iaQ_dounble_prime)
        F += 0.5 * contract('me,ie->mi', Fme, t1)
        return F

    def build_Fae(self, tQ, BovQ, BvvQ, Tau_iaQ_prime, t1, Fme):
        contract = self.contract
        F = contract('aeQ,Q->ae', BvvQ, tQ)
        F -= contract('maQ,meQ->ae', Tau_iaQ_prime, BovQ)
        F -= 0.5 * contract('ma,me->ae', t1, Fme)
        return F

    def build_Xijab(self, t2, Fae, Fmi, tiaQ_prime, tiaQ, taiQ, BovQ):
        #start = time.time()
        contract = self.contract
        X = contract('ijae,be->ijab', t2, Fae)
        X -= contract('mjab,mi->ijab', t2, Fmi)
        X += contract('iaQ,jbQ->ijab', tiaQ_prime, BovQ)
        X -= contract('aiQ,jbQ->ijab', taiQ, tiaQ)

        return X + X.swapaxes(0,1).swapaxes(2,3)

    def _idx2(self, i, j):
        return i * (i + 1) // 2 + j

    def build_WmnijT2(self, BooQ, BovQ, tijQ, tau_ijab):
        #start = time.time()
        contract = self.contract
        W = contract("miQ,njQ->minj", BooQ, BooQ).swapaxes(1, 2)
        X = contract("imQ,jnQ->mnij", tijQ, BooQ)
        W += X + X.swapaxes(0,1).swapaxes(2,3)
        V = contract("meQ,nfQ->menf", BovQ, BovQ).swapaxes(1, 2)
        W += contract("mnef,ijef->mnij", V, tau_ijab)
        W1 = contract("mnij,mnab->ijab", W, tau_ijab)

        return W1

    def build_WmnijT2_packed(self, no, nv, BooQ, BovQ, tijQ, tau_ijab):
        """
        Packed-symmetric / antisymmetric version of WmnijT2 that reproduces:

            W_full = <mn|ij> + X + X^(mn↔nm,ij↔ji) + Σ_ef (mn|ef) tau_ij^ef

        and returns  Σ_{mnab} W_full(mn,ij) tau_mn^ab as in build_WmnijT2.
        """
        #start = time.time()
        contract = self.contract

        # 0. <mn|ij>
        W = contract("miQ,njQ->minj", BooQ, BooQ).swapaxes(1, 2)  # (m,n,i,j)

        # 1. X-part: X + X_perm, reshuffled to (m,n,i,j)
        X = contract("imQ,jnQ->imjn", tijQ, BooQ)  # (i,m,j,n)
        X += X.swapaxes(0, 2).swapaxes(1, 3)  # X + X^(i↔j,m↔n)
        X = X.swapaxes(0, 1).swapaxes(1, 3).swapaxes(2, 3)  # → (m,n,i,j)

        # 2. (mn|ef) = sum_Q B_me^Q B_nf^Q
        V = contract("meQ,nfQ->menf", BovQ, BovQ).swapaxes(1, 2)  # (m,n,e,f)

        ntri_oo = no * (no + 1) // 2
        ntri_vv = nv * (nv + 1) // 2

        # symmetric / antisymmetric in (e,f) with (2 - δ_ef) normalization
        Vs = np.zeros((ntri_oo, ntri_vv))
        Va = np.zeros((ntri_oo, ntri_vv))

        for m in range(no):
            for n in range(m + 1):
                mn_p = self._idx2(m, n)
                for e in range(nv):
                    for f in range(e + 1):
                        ef_p = self._idx2(e, f)
                        V_mnef = V[m, n, e, f]
                        if e == f:
                            # diagonal: only one element (ef == fe)
                            Vs[mn_p, ef_p] = V_mnef
                            Va[mn_p, ef_p] = 0.0
                        else:
                            V_mnfe = V[m, n, f, e]
                            # C++ packed convention: multiply by (2 - δ_ef) = 2 for off-diagonal
                            Vs[mn_p, ef_p] = 0.5 * (V_mnef + V_mnfe)  # symmetric
                            Va[mn_p, ef_p] = 0.5 * (V_mnef - V_mnfe)  # antisymmetric

        # 3. Pack Tau_ijab in ij and ab with (2 - δ_ab) as in C++
        Tau_p = np.zeros((ntri_oo, ntri_vv))
        Tau_a = np.zeros((ntri_oo, ntri_vv))

        for i in range(no):
            for j in range(i + 1):
                ij_p = self._idx2(i, j)
                for a in range(nv):
                    for b in range(a + 1):
                        ab_p = self._idx2(a, b)
                        tau_ijab_val = tau_ijab[i, j, a, b]
                        tau_jiab_val = tau_ijab[j, i, a, b]

                        # same (2 - δ_ab) normalization used in the C++ packed code
                        w_ab = 2.0 if a != b else 1.0

                        Tau_p[ij_p, ab_p] = 0.5 * (tau_ijab_val + tau_jiab_val) * w_ab
                        Tau_a[ij_p, ab_p] = 0.5 * (tau_ijab_val - tau_jiab_val) * w_ab

        # 4. Contract in packed space:
        #    S(mn,ij)  = Σ_{a>=b} Vs(mn,ab) * Tau_p(ij,ab)
        #    A(mn,ij)  = Σ_{a>=b} Va(mn,ab) * Tau_a(ij,ab)
        S_packed = Vs @ Tau_p.T  # (mn_pair, ij_pair)
        A_packed = Va @ Tau_a.T

        # 5. Unpack to full W_tau(m,n,i,j)
        W_tau = np.zeros((no, no, no, no))

        for m in range(no):
            for n in range(no):
                if m >= n:
                    mn_p = self._idx2(m, n)
                    s_mn = 1.0
                else:
                    mn_p = self._idx2(n, m)
                    s_mn = -1.0

                for i in range(no):
                    for j in range(no):
                        if i >= j:
                            ij_p = self._idx2(i, j)
                            s_ij = 1.0
                        else:
                            ij_p = self._idx2(j, i)
                            s_ij = -1.0

                        # combine (+) and (−) parts with ij/mn antisymmetry
                        W_tau[m, n, i, j] = S_packed[mn_p, ij_p] + (s_mn * s_ij) * A_packed[mn_p, ij_p]

        # 6. Assemble W = <mn|ij> + X + W_tau and contract with full tau_ijab
        W += X + W_tau
        W1 = contract("mnij,mnab->ijab", W, tau_ijab)
        #print("time taken to compute WmnijT2_packed: ", time.time() - start)

        return W1

    def build_WmbejT2(self, BovQ, BooQ, BvvQ, tijQ, tabQ, tiaQ_prime, TiaQ, t2, u_ijab):
        #start = time.time()
        contract = self.contract
        # W(me,jb)
        W1 = contract('meQ,jbQ->mejb', BovQ, BovQ)
        tmp = tiaQ_prime + 0.5 * TiaQ
        W1 += contract('meQ,jbQ->mejb', BovQ, tmp)
        V = contract('mfQ,neQ->mfne', BovQ, BovQ)
        W1 -= 0.5 * contract('mfne,jnbf->mejb', V, t2)
        # W'(me,jb)
        W2 = contract('mjQ,ebQ->mejb', BooQ, BvvQ)
        W2 -= contract('jmQ,beQ->mejb', tijQ + BooQ, tabQ)
        W2 += contract('jmQ,beQ->mejb', tijQ, BvvQ)
        W2 -= 0.5 * contract('mfne,njbf->mejb', V, t2)
        # t_ij^ab <- P_ij^ab[0.5 * C(ia,jb) + C(ib,ja)]
        C = -contract('miae,mejb->ijab', t2, W2)
        tmpC = 0.5 * C + C.swapaxes(2,3)
        W_T2 = tmpC + tmpC.swapaxes(0,1).swapaxes(2,3)
        # t_ij^ab <- P_ij^ab Dijab
        D = 0.5 * contract('imae,mejb->ijab', u_ijab, 2 * W1 - W2)
        W_T2 += D + D.swapaxes(0,1).swapaxes(2,3)

        #print("time taken to compute WmbejT2: ", time.time() - start)
        return W_T2

    def build_WijamT2(self, BvvQ, BovQ, t1, tau_ijab):
        #start = time.time()
        contract = self.contract
        V = contract("aeQ,mfQ->aemf", BvvQ, BovQ).swapaxes(1,2)
        #V = contract("amQ,efQ->amef", BvvQ, BovQ).swapaxes(1, 2)
        W = contract("ijef,amef->ijam", tau_ijab, V)
        R = -1.0 * contract("ijam,mb->ijab", W, t1)
        #print("time taken to compute WijamT2: ", time.time() - start)

        return R + R.swapaxes(0,1).swapaxes(2,3)

    def build_WijamT2_packed(self, no, nv, BvvQ, BovQ, t1, tau_ijab):
        """
        Packed version of the Wijam^(T2) kernel that matches build_WijamT2.

        Parameters
        ----------
        tau_ijab : (nocc, nocc, nvir, nvir)
            Tau_{ij}^{ef}
        BvvQ : (nvir, nvir, nQ)
            B_{ae}^Q
        BovQ : (nocc, nvir, nQ)
            B_{mf}^Q

        Returns
        -------
        X + X^P : (nocc, nocc, nvir, nvir)
            Contribution to R_{ijab} from the T2 kernel, with ij/ab antisymmetrized.
        """
        #start = time.time()
        contract = self.contract

        ntri_oo = no * (no + 1) // 2
        ntri_vv = nv * (nv + 1) // 2

        # ------------------------------------------------------------------
        # 1. Pack Tau_{ij}^{ef} over ij (±) and ef (with 2-δ_ef weighting)
        # ------------------------------------------------------------------
        Tau_s = np.zeros((ntri_oo, ntri_vv))  # symmetric in ij
        Tau_a = np.zeros((ntri_oo, ntri_vv))  # antisymmetric in ij

        for i in range(no):
            for j in range(i + 1):
                ijp = self._idx2(i, j)
                for e in range(nv):
                    for f in range(e + 1):
                        efp = self._idx2(e, f)
                        t_ij_ef = tau_ijab[i, j, e, f]
                        t_ji_ef = tau_ijab[j, i, e, f]

                        # weight for packing e>=f: off-diagonals appear twice in the full sum
                        w_ef = 2.0 if e != f else 1.0

                        Tau_s[ijp, efp] = 0.5 * (t_ij_ef + t_ji_ef) * w_ef
                        Tau_a[ijp, efp] = 0.5 * (t_ij_ef - t_ji_ef) * w_ef

        # ------------------------------------------------------------------
        # 2. allocate output: W_{ijam}
        # ------------------------------------------------------------------
        Wijam = np.zeros((no, no, nv, no))

        # temporary for S(a,ij) and A(a,ij) for a fixed m
        S = np.zeros((nv, ntri_oo))
        A = np.zeros((nv, ntri_oo))

        # ------------------------------------------------------------------
        # 3. loop over m
        #    I_m(a,e,f) = Σ_Q B_{ae}^Q B_{mf}^Q
        # ------------------------------------------------------------------
        for m in range(no):
            B_mfQ = BovQ[m, :, :]  # (f,Q)

            # I_m[a,e,f]
            I_m = contract("aeQ,fQ->aef", BvvQ, B_mfQ)  # (a,e,f)

            # pack over (e,f) → Vp, Va
            Vp = np.zeros((nv, ntri_vv))  # symmetric in ef
            Va = np.zeros((nv, ntri_vv))  # antisymmetric in ef

            for a in range(nv):
                for e in range(nv):
                    for f in range(e + 1):
                        efp = self._idx2(e, f)
                        val_ef = I_m[a, e, f]
                        if e == f:
                            # diagonal: antisymmetric part is zero
                            Vp[a, efp] = val_ef
                            Va[a, efp] = 0.0
                        else:
                            val_fe = I_m[a, f, e]
                            Vp[a, efp] = 0.5 * (val_ef + val_fe)
                            Va[a, efp] = 0.5 * (val_ef - val_fe)

            # 4. contract packed ef with packed Tau over ef
            #    S[a,ij] = Σ_{e>=f} Vp[a,ef] * Tau_s[ij,ef]
            #    A[a,ij] = Σ_{e>=f} Va[a,ef] * Tau_a[ij,ef]
            S[:] = Vp @ Tau_s.T  # (a, ij_pair)
            A[:] = Va @ Tau_a.T

            # 5. unpack back to Wijam(i,j,a,m)
            for a in range(nv):
                for i in range(no):
                    for j in range(no):
                        if i >= j:
                            ijp = self._idx2(i, j)
                            s_ij = 1.0
                        else:
                            ijp = self._idx2(j, i)
                            s_ij = -1.0
                        Wijam[i, j, a, m] = S[a, ijp] + s_ij * A[a, ijp]

        # ------------------------------------------------------------------
        # 6. contract with t1_{mb} and antisymmetrize in ij and ab
        # ------------------------------------------------------------------
        X = -contract("ijam,mb->ijab", Wijam, t1)
        #print("time taken to compute WijamT2_packed: ", time.time() - start)

        return X + X.swapaxes(0, 1).swapaxes(2, 3)

    def build_WabefT2(self, BvvQ, tau_ijab):
        #start = time.time()
        contract = self.contract
        V = contract('aeQ,bfQ->aebf', BvvQ, BvvQ).swapaxes(1,2)
        W = contract('ijef,abef->ijab', tau_ijab, V)
        #print("time taken to compute WabefT2: ", time.time() - start)

        return W

    def build_WabefT2_packed(self, no, nv, BvvQ, tau_ijab):
        """
        DF-packed version of the W_abef^(T2)-like kernel that matches build_WabefT2.

        Parameters
        ----------
        BvvQ : (nvir, nvir, nQ)
            B_{ae}^Q  (virtual, virtual, Q)
        tau_ijab : (nocc, nocc, nvir, nvir)
            Tau_{ij}^{ef}

        Returns
        -------
        W : (nocc, nocc, nvir, nvir)
            W_{ijab} from the T2 kernel.
        """
        #start1 = time.time()
        contract = self.contract

        # packed counts
        ntri_vv = nv * (nv + 1) // 2
        ntri_oo = no * (no + 1) // 2

        # ------------------------------------------------------------------
        # 1. pack Tau_{ij}^{ef} into (+) and (-) with (2 - δ_ef) weight
        # ------------------------------------------------------------------
        Tau_s = np.zeros((ntri_oo, ntri_vv))  # (+)
        Tau_a = np.zeros((ntri_oo, ntri_vv))  # (-)

        for i in range(no):
            for j in range(i + 1):
                ij_p = self._idx2(i, j)
                for e in range(nv):
                    for f in range(e + 1):
                        ef_p = self._idx2(e, f)
                        t_ij_ef = tau_ijab[i, j, e, f]
                        t_ji_ef = tau_ijab[j, i, e, f]

                        # off-diagonal (e≠f) appear twice in the full sum over e,f
                        w_ef = 2.0 if e != f else 1.0

                        Tau_s[ij_p, ef_p] = 0.5 * (t_ij_ef + t_ji_ef) * w_ef
                        Tau_a[ij_p, ef_p] = 0.5 * (t_ij_ef - t_ji_ef) * w_ef

        # ------------------------------------------------------------------
        # 2. allocate S and A in packed (ab, ij)-space
        # ------------------------------------------------------------------
        S_packed = np.zeros((ntri_vv, ntri_oo))
        A_packed = np.zeros((ntri_vv, ntri_oo))

        # ------------------------------------------------------------------
        # 3. for each a, build V[a](b,e,f) = sum_Q B_{bf}^Q B_{ae}^Q,
        #    then split over (e,f) into (+)/(-) and contract with Tau_s/Tau_a
        # ------------------------------------------------------------------
        for a in range(nv):
            # V_a_bfe_raw[b, f, e] = Σ_Q B_{bf}^Q B_{ae}^Q
            V_a_bfe_raw = contract('bfQ,eQ->bfe', BvvQ, BvvQ[a, :, :])  # (b,f,e)

            # make V_a_bef[b,e,f] for convenience
            V_a_bef = V_a_bfe_raw.transpose(0, 2, 1)  # (b,e,f)

            # split into (+) and (-) over (e,f)
            Vp = np.zeros((nv, ntri_vv))  # (b, e>=f)   symmetric
            Va = np.zeros((nv, ntri_vv))  # (b, e>=f)   antisymmetric

            for b in range(nv):
                for e in range(nv):
                    for f in range(e + 1):
                        ef_p = self._idx2(e, f)
                        v_bef = V_a_bef[b, e, f]
                        if e == f:
                            # diagonal: antisymmetric component must vanish
                            Vp[b, ef_p] = v_bef
                            Va[b, ef_p] = 0.0
                        else:
                            v_bfe = V_a_bef[b, f, e]
                            Vp[b, ef_p] = 0.5 * (v_bef + v_bfe)
                            Va[b, ef_p] = 0.5 * (v_bef - v_bfe)

            # contract with packed Tau:
            # Ts[b, ij] = sum_{e>=f} Vp[b, ef] * Tau_s[ij, ef]
            # Ta[b, ij] = sum_{e>=f} Va[b, ef] * Tau_a[ij, ef]
            Ts = Vp @ Tau_s.T  # (b, ij_pair)
            Ta = Va @ Tau_a.T  # (b, ij_pair)

            # accumulate into S_packed, A_packed under ab with b <= a
            for b in range(a + 1):
                ab_p = self._idx2(a, b)
                S_packed[ab_p, :] += Ts[b, :]
                A_packed[ab_p, :] += Ta[b, :]

        # ------------------------------------------------------------------
        # 4. unpack back to full W_{ijab}
        # ------------------------------------------------------------------
        W = np.zeros((no, no, nv, nv))

        for a in range(nv):
            for b in range(nv):
                if a >= b:
                    ab_p = self._idx2(a, b)
                    s_ab = 1.0
                else:
                    ab_p = self._idx2(b, a)
                    s_ab = -1.0

                for i in range(no):
                    for j in range(no):
                        if i >= j:
                            ij_p = self._idx2(i, j)
                            s_ij = 1.0
                        else:
                            ij_p = self._idx2(j, i)
                            s_ij = -1.0

                        val = S_packed[ab_p, ij_p] + (s_ab * s_ij) * A_packed[ab_p, ij_p]
                        W[i, j, a, b] = val
        #print("time taken to compute WabefT2_packed: ", time.time() - start1)

        return W

    def r_T1(self, t1, Fmi, Fme, Fae, u_ijab, tQ, TiaQ, tiaQ, BooQ, BovQ, BvvQ):
        #start = time.time()
        contract = self.contract
        # Fia, first term in Ria(Eq.19)
        r1 = contract('ae,ie->ia', Fae, t1)
        r1 -= contract('mi,ma->ia', Fmi, t1)
        # Cia(Eq.22)
        r1 += contract('imae,me->ia', u_ijab, Fme)
        # Aia(Eq.20)
        r1 += contract('Q,iaQ->ia', tQ, BovQ)
        # Bia(Eq.21)
        r1 -= contract('imQ, maQ->ia', BooQ, TiaQ + tiaQ)
        # Aia(Eq.20)
        r1 += contract('ieQ,eaQ->ia', TiaQ, BvvQ)
        
        return r1

    def r_T2(self, Xijab, BovQ, WmnijT2, WmbejT2, WijamT2, WabefT2):
        #start = time.time()
        contract = self.contract
        # first term in Eijab(Eq.15) + first term in Gijab(Eq.16) + part of first term in Rijab(Eq.10)
        r2 = Xijab.copy()
        # part of first term in Rijab(Eq.10) with bare BovQ
        r2 += contract('iaQ,jbQ->ijab', BovQ, BovQ)
        # Bijab(Eq.12)
        r2 += WmnijT2
        # Cijab(Eq.13) + Dijab(Eq.14)
        r2 += WmbejT2
        # part of Gijab(Eq.16)
        r2 += WijamT2
        # part of Eijab(Eq.15) + Aijab(Eq.11)
        r2 += WabefT2
        #print("time taken to build r2: ", time.time() - start)
        
        return r2

    def cc_energy(self, o, v, F, BovQ, t1, t2):
        contract = self.contract

        ecc = 2.0 * contract('ia,ia->', F[o,v],t1)
        tau = self.build_Tau_ijab(t1, t2)
        tau = 2 * tau - tau.swapaxes(2,3)
        V = contract('iaQ, jbQ->ijab', BovQ, BovQ)
        ecc += contract('ijab, ijab->', tau, V)

        return ecc

    def P_ijkabc(self, X):
        assert X.ndim == 6, "X must be a rank-6 tensor (i,j,k,a,b,c)."

        result = np.zeros_like(X)

            # Permute the first 3 indices (i,j,k) and apply the same permutation
            # to the last 3 indices (a,b,c)
        for p in itertools.permutations(range(3)):  # p is a tuple like (0,1,2), (1,2,0), ...
            order = p + tuple(i + 3 for i in p)  # e.g. (0,1,2,3,4,5) or (1,2,0,4,5,3)
            result += X.transpose(order)

        return result

    def triples_correction_1(self, o, v, F, BovQ, BooQ, BvvQ, t1, t2, Dijkabc):
        contract  = self.contract
        # Calculate 4-index integrals
        Vovvv = contract('iaQ,ebQ->iaeb', BovQ, BvvQ)
        Vooov = contract('mjQ,kcQ->mjkc', BooQ, BovQ)
        Vovov = contract('jbQ,kcQ->jbkc', BovQ, BovQ)
        # Build W
        W = contract('iaeb,jkec->ijkabc', Vovvv, t2)
        W -= contract('mjkc,imab->ijkabc', Vooov, t2)
        Wijkabc = self.P_ijkabc(W)

        #Wijkabc = W.copy()
        # Build V
        Fme = F[o,v].copy()
        Vijkabc = Wijkabc.copy()
        Vijkabc += contract('ia,jbkc->ijkabc', t1, Vovov)
        Vijkabc += contract('jb,iakc->ijkabc', t1, Vovov)
        Vijkabc += contract('kc,iajb->ijkabc', t1, Vovov)
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

        et = 1 / 3 * contract('ijkabc,ijkabc->', W_tmp, V_tmp)

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
        start = time.time()
        contract = self.contract
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
        W = contract('iaeb,jkec->ijkabc', Vovvv, t2)
        W -= contract('mjkc,imab->ijkabc', Vooov, t2)

        # Apply permutation operator P_{ijk}^{abc} so W is symmetrized
        # in (i,j,k) and (a,b,c), matching the derivation in the paper.
        W = self.P_ijkabc(W)

        # --------------------------------------------------------------
        # 3. Build V_{ijk}^{abc} from W (again mirroring your working code)
        #    V = W + t1 * (ov|ov)  (+ optional F·t2 terms if you want them)
        # --------------------------------------------------------------
        Fme = F[o, v].copy()

        V = W.copy()
        V += contract('ia,jbkc->ijkabc', t1, Vovov)
        V += contract('jb,iakc->ijkabc', t1, Vovov)
        V += contract('kc,iajb->ijkabc', t1, Vovov)

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
        print("time taken to calculate (T) in seconds: ", time.time() - start)
        return E_T

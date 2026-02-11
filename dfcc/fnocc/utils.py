import numpy as np
import einsums as ein

class helper_diis:
    def __init__(self, max_diis):
        self.max_diis = max_diis
        self.vecs_t1 = []
        self.vecs_t2 = []
        self.errs = []

    def add_error_vector(self, t1_vec, t2_vec, dt1, dt2):
        # store amplitudes (oldvector in C++)
        self.vecs_t1.append(t1_vec.copy())
        self.vecs_t2.append(t2_vec.copy())

        # store error/update (evector in C++)
        e = np.concatenate((dt2.ravel(), dt1.ravel()))  # match C++ order: t2 then t1
        self.errs.append(e)

        # truncate
        if len(self.errs) > self.max_diis:
            self.errs.pop(0)
            self.vecs_t1.pop(0)
            self.vecs_t2.pop(0)

    def extrapolate(self):
        m = len(self.errs)
        if m == 0:
            raise RuntimeError("No DIIS vectors stored.")

        B = np.empty((m + 1, m + 1))
        B[:-1, :-1] = [[np.dot(self.errs[i], self.errs[j]) for j in range(m)] for i in range(m)]
        B[:-1, -1] = -1.0
        B[-1, :-1] = -1.0
        B[-1, -1] = 0.0

        rhs = np.zeros(m + 1)
        rhs[-1] = -1.0

        # IMPORTANT: do NOT normalize B (see Fix #2)
        c = np.linalg.solve(B, rhs)[:-1]

        t1 = np.zeros_like(self.vecs_t1[0])
        t2 = np.zeros_like(self.vecs_t2[0])
        for ci, v1, v2 in zip(c, self.vecs_t1, self.vecs_t2):
            t1 += ci * v1
            t2 += ci * v2

        return t1, t2

# FNO: virtual-virtual block of one-particle density matrix
class fnocc(object):
    def __init__(self, Dijab, tolerance):
        self.tolerance = tolerance
        self.Dijab = Dijab # Dijab from non-T1-transformed Dijab

    def opdm_vv(self, contract, BovQ):
    # gamma_ab = 2 * sum_{ijc} [([2(ia|jc) - (ic|ja)](ib|jc)) / D_ijac * D_ijbc]
        # build 4-index integrals
        # (ia|jc)
        Vovov = contract("iaQ,jcQ->iajc", BovQ, BovQ)
        # 2 * ([2 * (ia|jc) - (ic|ja)](ib|jc))
        gamma_ab = contract("iajc,ibjc->ab", 2 * Vovov - Vovov.swapaxes(1,3), Vovov, alpha=2.0)
        # gamma_ab /= (F_ii + F_jj - F_aa - F_cc)(F_ii + F_jj - F_bb - F_cc)
        Dijab_prime = contract("ijac,ijbc->ab", self.Dijab, self.Dijab)
        gamma_ab /= Dijab_prime
        return gamma_ab

    def natural_orbs(self, gamma_ab):
        # occupation numbers
        diag = np.diag(gamma_ab)
        print(diag, diag.shape)
        # count number of natural virtual orbitals
        count = 0
        for i in range(diag.shape[0]):
            if diag[i] >= self.tolerance:
                count += 1
            else:
                break
        return count

# Wrapper function for Einsums tensor contraction
class einsums_contract(object):
    def parse_spec(self, spec: str):
        spec = spec.replace(" ", "")
        lhs, out = spec.split("->")
        a, b = lhs.split(",")
        return a, b, out

    def out_shape(self, out_idx, a_idx, A, b_idx, B):
        dim_map = {}

        if len(a_idx) != A.ndim:
            raise ValueError(f"Index '{a_idx}' has length {len(a_idx)}, but A has ndim {A.ndim}")
        for idx, dim in zip(a_idx, A.shape):
            if idx in dim_map and dim_map[idx] != dim:
                raise ValueError(f"Dimension mismatch for index '{idx}': {dim_map[idx]} vs {dim}")
            dim_map[idx] = dim

        if len(b_idx) != B.ndim:
            raise ValueError(f"Index '{b_idx}' has length {len(b_idx)}, but B has ndim {B.ndim}")
        for idx, dim in zip(b_idx, B.shape):
            if idx in dim_map and dim_map[idx] != dim:
                raise ValueError(f"Dimension mismatch for index '{idx}': {dim_map[idx]} vs {dim}")
            dim_map[idx] = dim

        try:
            return tuple(dim_map[idx] for idx in out_idx)
        except KeyError as e:
            raise ValueError(f"Unknown index '{e.args[0]}' in output spec '{out_idx}'")

    def allocator(self, shape, like):
        return np.zeros(shape, dtype=getattr(like, "dtype", None))

    def _einsums_execute(self, out_idx, a_idx, b_idx, out, A, B, alpha, beta):
        """
        Einsums execute convention in your code:
          plan.execute(beta, out, alpha, A, B)
        """
        plan = ein.core.compile_plan(out_idx, a_idx, b_idx)
        # print(plan)  # keep if you want
        plan.execute(beta, out, alpha, A, B)
        return out

    def _packable_gemm(self, a_idx, b_idx, out_idx):
        """
        Recognize a GEMM-able 2-tensor contraction:
          - contracted indices are those appearing in both inputs but not in output
          - output indices must be exactly [A_free indices in A order] + [B_free indices in B order]
        This matches your working example: mnef,ijef->mnij
          A_free = mn (in A order)
          B_free = ij (in B order)
          contracted = ef
          out = mnij (A_free + B_free)
        """
        a_set = set(a_idx)
        b_set = set(b_idx)
        out_set = set(out_idx)

        shared = a_set & b_set
        contracted = [x for x in a_idx if (x in shared and x not in out_set)]
        # free indices in each operand, in operand order
        a_free = [x for x in a_idx if x not in contracted]
        b_free = [x for x in b_idx if x not in contracted]

        # sanity: output must be exactly a_free followed by b_free
        if list(out_idx) != a_free + b_free:
            return None

        # contracted must appear in both
        if any(x not in b_set for x in contracted):
            return None

        return a_free, b_free, contracted

    def _moveaxes_reshape(self, X, idx, first_group, second_group):
        """
        Reorder axes of X so that axes for first_group come first, then second_group,
        then reshape into (prod(first_group dims), prod(second_group dims)).
        """
        idx_list = list(idx)
        perm = [idx_list.index(ch) for ch in (first_group + second_group)]
        Xp = np.moveaxis(X, perm, list(range(len(perm))))
        s = Xp.shape
        d1 = int(np.prod(s[:len(first_group)], dtype=np.int64)) if first_group else 1
        d2 = int(np.prod(s[len(first_group):len(first_group)+len(second_group)], dtype=np.int64)) if second_group else 1
        X2 = Xp.reshape(d1, d2)
        return X2

    def contract(self, spec: str, A, B, *, alpha: float = 1.0, beta: float = 0.0, out=None, prefer_gemm=True):
        a_idx, b_idx, out_idx = self.parse_spec(spec)

        # allocate output if needed
        if out is None:
            shape = self.out_shape(out_idx, a_idx, A, b_idx, B)
            if out_idx == "" and shape == ():
                shape = (1,)
            out = self.allocator(shape, A)

        # If user provides out, we assume they want it filled in-place.
        # We'll reshape out as needed for GEMM path but keep same buffer.

        # Try GEMM packing path
        pack = self._packable_gemm(a_idx, b_idx, out_idx) if prefer_gemm else None
        if pack is not None:
            a_free, b_free, contracted = pack

            # Build 2D A2 = (a_free, contracted)
            A2 = self._moveaxes_reshape(A, a_idx, a_free, contracted)
            # Build 2D B2 = (b_free, contracted)
            B2 = self._moveaxes_reshape(B, b_idx, b_free, contracted)

            # We want C2 = A2 @ B2.T. Einsums GEMM-friendly: "ij", "ik", "kj"
            # Let i = a_free compound, j = b_free compound, k = contracted compound
            # So pass A2 as (i,k) and BT as (k,j)
            A2 = np.ascontiguousarray(A2)
            BT = np.ascontiguousarray(B2.T)

            # Reshape output buffer to 2D view (i,j) matching out_idx = a_free + b_free
            # First ensure out is contiguous because Einsums GEMM tends to assume this.
            out_contig = out if out.flags['C_CONTIGUOUS'] else np.ascontiguousarray(out)

            # If we copied due to contiguity, we must write back at end.
            out_was_view = (out_contig is out)

            out2 = out_contig.reshape(
                int(np.prod([A.shape[list(a_idx).index(ch)] for ch in a_free], dtype=np.int64)) if a_free else 1,
                int(np.prod([B.shape[list(b_idx).index(ch)] for ch in b_free], dtype=np.int64)) if b_free else 1,
            )

            # Execute GEMM plan
            # out2 = beta*out2 + alpha*(A2 @ BT)
            self._einsums_execute("ij", "ik", "kj", out2, A2, BT, alpha, beta)

            # Reshape back to original out shape
            out_final = out2.reshape(self.out_shape(out_idx, a_idx, A, b_idx, B))

            if not out_was_view:
                # write results back into provided out buffer
                out[...] = out_final
                return out

            return out_final

        # Fallback: direct einsums (may be wrong/slow depending on library state)
        # Ensure contiguity for safety (does not guarantee correctness if planner is buggy)
        A_in = A
        B_in = B
        out_in = out
        self._einsums_execute(out_idx, a_idx, b_idx, out_in, A_in, B_in, alpha, beta)

        if out is not None and out.size == 1 and out_idx == "":
            return out.item(0)
        return out

    def __call__(self, spec, A, B, *, alpha=1.0, beta=0.0, out=None, prefer_gemm=True):
        return self.contract(spec, A, B, alpha=alpha, beta=beta, out=out, prefer_gemm=prefer_gemm)




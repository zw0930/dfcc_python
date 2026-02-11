import numpy as np
import einsums as ein

class helper_diis(object):
    def __init__(self, t1, t2, max_diis):

        self.oldt1 = t1.copy()
        self.oldt2 = t2.copy()
        self.diis_vals_t1 = [t1.copy()]
        self.diis_vals_t2 = [t2.copy()]

        self.diis_errors = []
        self.diis_size = 0
        self.max_diis = max_diis

    def add_error_vector(self, t1, t2):
        # Add DIIS vectors
        self.diis_vals_t1.append(t1.copy())
        self.diis_vals_t2.append(t2.copy())
        # Add new error vectors
        error_t1 = (self.diis_vals_t1[-1] - self.oldt1).ravel()
        error_t2 = (self.diis_vals_t2[-1] - self.oldt2).ravel()
        self.diis_errors.append(np.concatenate((error_t1, error_t2)))
        self.oldt1 = t1.copy()
        self.oldt2 = t2.copy()

    def extrapolate(self, t1, t2):

        if (self.max_diis == 0):
            return t1, t2

        # Limit size of DIIS vector
        if (len(self.diis_errors) > self.max_diis):
            del self.diis_vals_t1[0]
            del self.diis_vals_t2[0]
            del self.diis_errors[0]

        self.diis_size = len(self.diis_errors)

        # Build error matrix B
        B = np.ones((self.diis_size + 1, self.diis_size + 1)) * -1
        B[-1, -1] = 0

        for n1, e1 in enumerate(self.diis_errors):
            B[n1, n1] = np.dot(e1, e1)
            for n2, e2 in enumerate(self.diis_errors):
                if n1 >= n2:
                    continue
                B[n1, n2] = np.dot(e1, e2)
                B[n2, n1] = B[n1, n2]

        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

        # Build residual vector
        resid = np.zeros(self.diis_size + 1)
        resid[-1] = -1

        # Solve pulay equations
        ci = np.linalg.solve(B, resid)

        # Calculate new amplitudes
        t1 = np.zeros_like(self.oldt1)
        t2 = np.zeros_like(self.oldt2)
        for num in range(self.diis_size):
            t1 += ci[num] * self.diis_vals_t1[num + 1]
            t2 += ci[num] * self.diis_vals_t2[num + 1]

        # Save extrapolated amplitudes to old_t amplitudes
        self.oldt1 = t1.copy()
        self.oldt2 = t2.copy()

        return t1, t2

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

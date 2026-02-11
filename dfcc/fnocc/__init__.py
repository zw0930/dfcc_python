from .ccwfn_fno import ccwfn as _ccwfn_ref
from .ccwfn_fno_ein import ccwfn as _ccwfn_ein

def ccwfn(*args, variant="ref", **kwargs):
    """
    Factory for FNO CC wavefunctions.

    Parameters
    ----------
    variant : str
        'ref' (default) : reference implementation using opt_einsum
        'ein'           : Einsums-based implementation
    """
    if variant == "ref":
        return _ccwfn_ref(*args, **kwargs)
    elif variant == "ein":
        return _ccwfn_ein(*args, **kwargs)
    else:
        raise ValueError(
            f"Unknown FNO CCWfn variant={variant!r}. "
            "Valid options: 'ref', 'ein'."
        )

__all__ = ["ccwfn"]


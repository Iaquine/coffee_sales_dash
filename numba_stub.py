"""
Numba stub for environments where numba is incompatible with the installed NumPy.
Provides minimal API surface used by shap: `njit` as a no-op decorator.
"""
import functools


def njit(func=None, **kwargs):
    """No-op replacement for numba.njit — returns the function unchanged."""
    if func is None:
        return lambda f: f
    return func


# Minimal stubs for other numba attributes imported transitively by shap
class _FakeModule:
    def __getattr__(self, name):
        return _noop

def _noop(*args, **kwargs):
    pass

typed = _FakeModule()
types = _FakeModule()
prange = range

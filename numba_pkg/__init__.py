"""Minimal numba stub package — no-op implementations for shap compatibility."""

def njit(func=None, **kwargs):
    if func is None:
        return lambda f: f
    return func

jit = njit
cfunc = njit
stencil = njit
prange = range

class typed:
    class List:
        pass
    class Dict:
        pass

class types:
    int64 = int
    float64 = float
    boolean = bool
    UniTuple = tuple
    Array = object
    def __getattr__(self, name):
        return object

class _FakeModule:
    def __getattr__(self, name):
        return _noop

def _noop(*args, **kwargs):
    pass

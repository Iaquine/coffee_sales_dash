"""Stub for numba.stencils"""
def stencil(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return lambda f: f

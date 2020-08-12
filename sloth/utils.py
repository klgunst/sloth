import numpy as np
import os
import ctypes

libname = 'libWigner.so'
try:
    libWigner = ctypes.cdll.LoadLibrary(libname)
except OSError:
    parentdir = os.path.dirname(os.path.realpath(__file__))
    libWigner = ctypes.cdll.LoadLibrary(os.path.join(parentdir, libname))

wigner3j = libWigner.wigner3j
wigner3j.argtypes = [ctypes.c_int] * 6
wigner3j.restype = ctypes.c_double

wigner6j = libWigner.wigner6j
wigner6j.argtypes = [ctypes.c_int] * 6
wigner6j.restype = ctypes.c_double

wigner9j = libWigner.wigner9j
wigner9j.argtypes = [ctypes.c_int] * 9
wigner9j.restype = ctypes.c_double


def flatten_svals(svals):
    """Flattens and sorts the singular values in a 1D array. It takes the
    multiplets into account.
    """
    # Gets all SU(2) symmetries
    su2ids = [ii for ii, s in enumerate(svals['symmetries']) if s == 'SU(2)']

    def unmultiplet(key, sval):
        # Undo the SU(2)
        multipl = np.prod([key[i] + 1 for i in su2ids])
        return np.repeat(sval, multipl)

    return np.sort(np.concatenate([unmultiplet(k, v) for k, v in svals.items()
                                   if isinstance(k, tuple)]))[::-1]


def renyi_entropy(svals, α=1):
    """Calculates the renyi entropy for a given dictionary of singular values.
    """
    sv = flatten_svals(svals)
    omega = sv ** 2
    is_zero = np.isclose(sv, np.zeros(sv.shape), atol=1e-32)
    if α == 1:
        V = -omega * np.log(omega, where=np.logical_not(is_zero))
        V[is_zero] = 0
        return np.sum(V)
    else:
        spomega = np.sum(np.power(omega, α))
        is_zero = np.isclose(spomega, np.zeros(spomega.shape), atol=1e-32)
        return np.log(spomega, where=np.logical_not(is_zero)) / (1 - α)

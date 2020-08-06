from sloth.tensor import Tensor, Leg
import sloth.symmetries as sls
import itertools
import numpy as np


rng = np.random.default_rng()


def make_random_l(nrirs, maxd):
    """Make legs with random dimensions
    """
    return Leg(), [(k, rng.integers(0, maxd, endpoint=True))
                   for k in itertools.product(*[range(x) for x in nrirs])]


def make_random_t(symmetr, legs, flow):
    """Make random tensors
    """
    A = Tensor(symmetr, [(l[0], f) for l, f in zip(legs, flow)])
    for x in itertools.product(*[l[1] for l in legs]):
        key = (tuple(y[0] for y in x),)
        shape = tuple(y[1] for y in x)
        if sls.is_allowed_coupling(key[0], flow, symmetr) and 0 not in shape:
            A[key] = rng.uniform(low=-1.0, high=1.0, size=shape)
    return A


def pytest_addoption(parser):
    parser.addoption("--onlyspin", action="store_true",
                     help="run only for SU(2)-adapted tensors")
    parser.addoption("--nospin", action="store_true",
                     help="run only for U(1)-adapted tensors")


def pytest_generate_tests(metafunc):
    if "kind" in metafunc.fixturenames:
        if metafunc.config.getoption("onlyspin"):
            kinds = ['SU(2)']
        elif metafunc.config.getoption("nospin"):
            kinds = ['U(1)']
        else:
            kinds = ['U(1)', 'SU(2)']
        metafunc.parametrize("kind", kinds)

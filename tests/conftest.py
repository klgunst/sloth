import numpy as np
import pytest


class Helpers:
    @staticmethod
    def close_paddy(x, y, **kwargs):
        """Compares two 1D-arrays and pads the shortest with zeros
        """
        maxlen = max(len(x), len(y))
        ar = [np.pad(z, (0, maxlen - len(z)), mode='constant') for z in (x, y)]
        return np.allclose(*ar, **kwargs)


@pytest.fixture
def helpers():
    return Helpers


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

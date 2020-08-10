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
    parser.addoption("--symmetries", nargs='+')


def pytest_generate_tests(metafunc):
    if "symmetr" in metafunc.fixturenames:
        if kinds := metafunc.config.getoption("symmetries"):
            kinds = [[k] for k in kinds]
        else:
            kinds = [['fermionic', 'U(1)'], ['SU(2)'], ['fermionic', 'SU(2)']]
        metafunc.parametrize("symmetr", kinds)

    if "kind" in metafunc.fixturenames:
        metafunc.parametrize("kind", ['SU(2)', 'U(1)'])

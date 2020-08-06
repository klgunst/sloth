from sloth.network import read_h5
from sloth.utils import flatten_svals
import numpy as np
import pickle
import pytest
import os


with open('N2_svals.pickle', 'rb') as f:
    checkvals = pickle.load(f)


def close_paddy(x, y, **kwargs):
    """Compares two 1D-arrays and pads the shortest with zeros
    """
    maxlen = max(len(x), len(y))
    ar = [np.pad(z, (0, maxlen - len(z)), mode='constant') for z in (x, y)]
    return np.allclose(*ar, **kwargs)


@pytest.mark.parametrize("netwType", ["DMRG"])
def test_singular_values(netwType):
    h5path = os.path.join(os.path.dirname(__file__), 'h5')
    svals = {}

    # Obtained from the T3NS C implementation (github.com/klgunst/T3NS)
    for kind in ['U1', 'SU2']:
        tns = read_h5(os.path.join(h5path, f'N2_{kind}_{netwType}.h5'))
        svals[kind] = {
            tuple(tuple(sorted(xx)) for xx in tns.get_orbital_partition(x)):
            flatten_svals(y) for x, y in
            tns.calculate_singular_values(tns.sink).items()
        }
        # Check it is the same as check value
        for x, y in checkvals[f'{kind}_{netwType}'].items():
            assert close_paddy(y, svals[kind][x])  # similar svals

    # Check that U1 is (approximately) the same as SU2
    for x, y in svals['U1'].items():
        assert np.isclose(np.linalg.norm(y), 1.)  # Normed tensor network
        assert close_paddy(y, svals['SU2'][x], atol=1e-6)  # similar svals

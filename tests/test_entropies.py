from sloth.network import read_h5
from sloth.utils import flatten_svals
import pickle
import pytest
import os


@pytest.fixture()
def checkvals():
    picklefile = os.path.join(os.path.dirname(__file__), 'N2_svals.pickle')
    with open(picklefile, 'rb') as f:
        checkvals = pickle.load(f)
    return checkvals


@pytest.mark.parametrize("netwType", ["DMRG", "T3NS"])
def test_singular_values(netwType, kind, checkvals, helpers):
    # Remove '(' and ')'
    kind = kind.replace(')', '').replace('(', '')
    h5path = os.path.join(os.path.dirname(__file__), 'h5')
    # Obtained from the T3NS C implementation (github.com/klgunst/T3NS)
    tns = read_h5(os.path.join(h5path, f'N2_{kind}_{netwType}.h5'))
    svals = {
        tuple(tuple(sorted(xx)) for xx in tns.get_orbital_partition(x)):
        flatten_svals(y) for x, y in
        tns.calculate_singular_values(tns.sink).items()
    }
    # Check it is the same as check value
    for x, y in checkvals[f'{kind}_{netwType}'].items():
        assert helpers.close_paddy(y, svals[x])  # similar svals

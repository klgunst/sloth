_SYMMETRIES = ['fermionic', 'U(1)', 'SU(2)', 'C1', 'Ci', 'C2', 'Cs', 'D2',
               'C2v', 'C2h', 'D2h', 'seniority']


def symswap(symmetry, key, permute):
    if symmetry not in _SYMMETRIES:
        raise ValueError(f'Invalid symmetry {symmetry}')

    if permute != [1, 0, 2]:
        raise ValueError('Only swapping of first two bonds for SU(2) '
                         f'allowed atm, not {permute}')

    if symmetry == 'fermionic':
        return 1 if sum(key[:2]) % 2 == 0 else -1
    elif symmetry == 'SU(2)':
        return 1 if sum(key) % 4 == 0 else -1
    else:
        return 1.

from sloth.symmetries import is_allowed_coupling, allowed_couplings


def test_is_allowed_coupling():
    """Testing several cases of allowed couplings.
    """

    symmetries = ['SU(2)', 'SU(2)', 'fermionic', 'U(1)', 'D2h']
    flow = [True, False, True]
    coupling = [
        [((1, 0, 1, -5, 0), (1, 2, 0, -7, 1), (2, 2, 1, -2, 0 ^ 1)), True],
        [((1, 0, 1, -5, 0), (1, 2, 0, -7, 1), (0, 2, 1, -2, 0 ^ 1)), True],
        [((1, 0, 1, -5, 0), (1, 2, 0, -7, 1), (0, 1, 1, -2, 0 ^ 1)), False],
        [((1, 0, 1, -5, 0), (1, 2, 0, -7, 1), (0, 2, 0, -2, 0 ^ 1)), False],
        [((1, 0, 1, -5, 0), (1, 2, 0, -7, 1), (0, 2, 1, 2, 0 ^ 1)), False],
        [((1, 0, 1, -5, 0), (1, 2, 0, -7, 1), (0, 2, 1, -2, 0 ^ 2)), False],
    ]

    for coupl, expected in coupling:
        assert is_allowed_coupling(coupl, flow, symmetries) is expected


def test_generate_allowed_couplings():
    symmetries = ['SU(2)', 'SU(2)', 'fermionic', 'U(1)', 'D2h']
    flow = [True, True, False]
    coupling = ((3, 5, 1, 5, 0), (2, 2, 1, 3, 1), None)

    allowed = set(
        ((1, 3, 0, 8, 0 ^ 1),
         (1, 5, 0, 8, 0 ^ 1),
         (1, 7, 0, 8, 0 ^ 1),
         (3, 3, 0, 8, 0 ^ 1),
         (3, 5, 0, 8, 0 ^ 1),
         (3, 7, 0, 8, 0 ^ 1),
         (5, 3, 0, 8, 0 ^ 1),
         (5, 5, 0, 8, 0 ^ 1),
         (5, 7, 0, 8, 0 ^ 1))
    )

    for newcp in allowed_couplings(coupling, flow, 2, symmetries):
        allowed.remove(newcp)

    assert allowed == set()

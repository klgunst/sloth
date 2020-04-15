def allowed_couplings(coupling, flow, free_id, symmetries):
    """Iterator over all the allowed Irreps for free_id in coupling if the
    other two couplings are fixed.
    """
    from itertools import product

    if len(coupling) != 3:
        raise ValueError(f'len(coupling) [{len(coupling)}] != 3')

    if len(flow) != 3:
        raise ValueError(f'len(flow) [{len(flow)}] != 3')

    other_ids = [0, 1, 2]
    other_ids.remove(free_id)
    other_c = [coupling[o] for o in other_ids]
    other_f = [flow[o] for o in other_ids]
    this_f = flow[free_id]

    def fermionic_constraint(oirr, oflow, tflow):
        yield sum(oirr) % 2

    def U1_constraint(oirr, oflow, tflow):
        sign = {True: 1, False: -1}
        yield sign[not tflow] * sum(sign[f] * x for x, f in zip(oirr, oflow))

    def pg_constraint(oirr, oflow, tflow):
        yield oirr[0] ^ oirr[1]

    def SU2_constraint(oirr, oflow, tflow):
        return range(abs(oirr[0] - oirr[1]), oirr[0] + oirr[1] + 1, 2)

    constraint = {
        'fermionic': fermionic_constraint,
        'U(1)': U1_constraint,
        'SU(2)': SU2_constraint,
        'seniority': U1_constraint,
        'C1': pg_constraint,
        'Ci': pg_constraint,
        'C2': pg_constraint,
        'Cs': pg_constraint,
        'D2': pg_constraint,
        'C2v': pg_constraint,
        'C2h': pg_constraint,
        'D2h': pg_constraint
    }

    for ncoupling in product(*[constraint[s](c, other_f, this_f)
                               for *c, s in zip(*other_c, symmetries)]):

        assert is_allowed_coupling([c if ii != free_id else ncoupling for ii, c
                                    in enumerate(coupling)], flow, symmetries)
        yield ncoupling


def is_allowed_coupling(coupling, flow, symmetries):
    """Return boolean if coupling is allowed.
    """
    if len(coupling) != 3:
        raise ValueError(f'len(coupling) [{len(coupling)}] != 3')

    if len(flow) != 3:
        raise ValueError(f'len(flow) [{len(flow)}] != 3')

    def fermionic_constraint(irreps, flow):
        return sum(irreps) % 2 == 0

    def U1_constraint(irreps, flow):
        sign = {True: 1, False: -1}
        return sum(sign[f] * x for x, f in zip(irreps, flow)) == 0

    def pg_constraint(irreps, flow):
        return irreps[0] ^ irreps[1] == irreps[2]

    def SU2_constraint(irreps, flow):
        return irreps[0] + irreps[1] >= irreps[2] and \
            abs(irreps[0] - irreps[1]) <= irreps[2] and sum(irreps) % 2 == 0

    constraint = {
        'fermionic': fermionic_constraint,
        'U(1)': U1_constraint,
        'SU(2)': SU2_constraint,
        'seniority': U1_constraint,
        'C1': pg_constraint,
        'Ci': pg_constraint,
        'C2': pg_constraint,
        'Cs': pg_constraint,
        'D2': pg_constraint,
        'C2v': pg_constraint,
        'C2h': pg_constraint,
        'D2h': pg_constraint
    }

    return False not in [constraint[s]([c[ii] for c in coupling], flow)
                         for ii, s in enumerate(symmetries)]


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

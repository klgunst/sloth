from sympy.physics.wigner import wigner_6j
import numpy as np
from functools import reduce


def _prefswap0(permute):
    # Fermionic prefactors for all different permutations
    def fpref_123(a, b, c):
        return 1.

    def fpref_213(a, b, c):
        # a and b are odd
        return -1. if a & 1 and b % 1 else 1.

    def fpref_132(a, b, c):
        # b and c are odd
        return -1. if b & 1 and c % 1 else 1.

    def fpref_321(a, b, c):
        # |1|(|2| + |3|) + |2||3|
        return -1. if (a * (b + c) + b * c) & 1 else 1.

    def fpref_312(a, b, c):
        # |3|(|1| + |2|)
        return -1. if c * (a + b) & 1 else 1.

    def fpref_231(a, b, c):
        # |1|(|2| + |3|)
        return -1. if a * (b + c) & 1 else 1.

    # SU(2) prefactors for different permutations
    def su2pref_even(a, b, c):
        # 1
        return 1.

    def su2pref_odd(a, b, c):
        # a + b + c
        return 1. if (a + b + c) % 4 == 0 else -1.

    prefdict = {}

    prefdict['fermionic'] = {
        (0, 1, 2): fpref_123,
        (1, 0, 2): fpref_213,
        (0, 2, 1): fpref_132,
        (2, 1, 0): fpref_321,
        (2, 0, 1): fpref_312,
        (1, 2, 0): fpref_231
    }[permute]

    prefdict['SU(2)'] = su2pref_even if permute == [0, 1, 2] or \
        permute == [1, 2, 0] or permute == [2, 0, 1] else su2pref_odd

    return prefdict


def _prefswap1(ll, il):
    """Prefactor for swapping of ll[0] and ll[1] between two couplings where
    the common leg is given by il[0] and il[1] between coupling 0 and 1.
    """
    # Fermionic prefactors for all different permutations
    # internal index in flattened
    ia, ib = il[0], il[1] + 3
    # swapped index in flattened
    sa, sb = ll[0], ll[1] + 3

    assert ia != sa
    assert ib != sb

    def fpref(okey, nkey):
        assert okey[ia] == okey[ib]
        assert nkey[ia] == nkey[ib]
        assert okey[sa] == nkey[sb]
        assert okey[sb] == nkey[sa]
        # Bring the internals next to each other as <x||x>
        parity = sum(okey[ia + 1:ib]) * okey[ia]

        # inbetweens, ignoring the removed internal legs
        inbet = sum(n for i, n in enumerate(nkey[sa + 1:sb], start=sa + 1)
                    if i not in (ia, ib))

        # Swap the two legs
        parity += inbet * (nkey[sb] + nkey[sa]) + nkey[sb] * nkey[sa]

        # Bring new internals back to original position
        parity += sum(nkey[ia + 1:ib]) * nkey[ia]
        return -1. if parity % 2 == 1 else 1.

    # Swapping is (j1, j2, j), (j3, j4, j) -> (j1, j3, j), (j2, j4, j)
    # cyclic permutations of each coupling allowed without prefactor,
    # acyclic permutations give raise to a prefactor.

    # Acyclic permutation if: (sa + 1) % 3 != ia or (sb - 1) % 3 != ib
    # Once this taken into account, use eq. 6j-swap from phd defense
    def su2acyclic(k):
        # Acyclic minus sign for (j1, j2, j3).
        assert sum(k) % 2 == 0
        return 1. if sum(k) % 4 == 0 else -1.

    pref_funcs = []
    # Acyclic permutation of first coupling
    if (sa + 1) % 3 != ia:
        pref_funcs.append(lambda x, y: su2acyclic(x[:3]) * su2acyclic(y[:3]))

    # Acyclic permutation of second coupling
    if (sb - 1) % 3 != ib:
        pref_funcs.append(lambda x, y: su2acyclic(x[3:]) * su2acyclic(y[3:]))

    ra = list(range(3))
    ra.remove(sa)
    ra.remove(ia)
    ra = ra[0]

    rb = list(range(3, 6))
    rb.remove(sb)
    rb.remove(ib)
    rb = rb[0]

    def su2pref(ok, nk, pref_funcs):
        w6j = wigner_6j(nk[sb] / 2., nk[rb] / 2., nk[ib] / 2.,
                        nk[sa] / 2., nk[ra] / 2., ok[ia] / 2.)
        pr = (nk[ia] + 1) * float(w6j) * \
            (1. if (nk[ia] + ok[ia] - ok[sa] - ok[sb]) % 4 == 0 else -1.)
        return reduce(lambda x, y: x * y, [p(ok, nk) for p in pref_funcs], pr)

    return {
        'fermionic': fpref,
        'SU(2)': lambda x, y: su2pref(x, y, pref_funcs)
    }


def _prefremoveSimpleLoop(loop, coupling):
    """
    For the calculation of the prefactor associated with the removal of a
    simple loop.

    Returns:
        dictionary with for each symmetry a function that calculates the
        associated prefactor.
    """
    # The flows of the legs in the couplings
    f1, f2 = [tuple(c[1] for c in coupling[x]) for x in loop['coupl']]
    # Check some things
    assert f1[loop['fid'][0]]
    assert not f2[loop['fid'][1]]
    assert f1[loop['lid'][0][0]] is not f2[loop['lid'][0][1]]
    assert f1[loop['lid'][1][0]] is not f2[loop['lid'][1][1]]

    # slices for fermionic signs
    # If the leg in the first coupling is outgoing, should overpass each other
    fsl, ssl = [slice(lid[0] + (not f1[lid[0]]), 3 + lid[1])
                for lid in loop['lid']]

    def fpref(key):
        key = list(key)
        parity = key[loop['lid'][0][0]] * sum(key[fsl])
        # Removing legs is same as setting their keys to zero
        key[loop['lid'][0][0]] = 0
        key[loop['lid'][0][1] + 3] = 0
        parity += key[loop['lid'][1][0]] * sum(key[ssl])
        return 1. if parity % 2 == 0 else -1.

    su2preff = [lambda k, x=x: 1. if k[x] % 2 == 0 else -1.
                for x, _ in loop['lid'] if f1[x]]

    # one reflect of coupling needed?
    assert (((loop['lid'][0][0] + 1) % 3 == loop['fid'][0])
            == ((loop['lid'][0][1] + 1) % 3 == loop['fid'][1])) \
        == (((loop['lid'][1][0] + 1) % 3 == loop['fid'][0])
            == ((loop['lid'][1][1] + 1) % 3 == loop['fid'][1]))

    if ((loop['lid'][0][0] + 1) % 3 == loop['fid'][0]) == \
            ((loop['lid'][0][1] + 1) % 3 == loop['fid'][1]):
        su2preff.append(lambda x: 1. if sum(x) % 4 == 0 else -1.)

    sign = tuple(1 if ii != loop['fid'][0] else -1 for ii in range(3))

    def su2ortho(k):
        # Orthogonality equation taking into account the insertion of a vacuum
        m = 1. / np.sqrt(k[loop['fid'][0]] + 1)
        return m if sum(s * x for s, x in zip(sign, k)) % 4 == 0 else -m
    su2preff.append(su2ortho)

    return {
        'fermionic': fpref,
        'SU(2)': lambda k: reduce(lambda x, y: x * y,
                                  [p(k[:3]) for p in su2preff])
    }


def _prefremovevac(vacid, flow):
    """Prefactor for removing a vacuum leg
    """
    fid, sid = [x for x in range(3) if x != vacid]
    assert flow[fid] != flow[sid]

    def fpref(k):
        return 1. if k % 2 == 0 or flow[fid] else -1.

    inid = fid if flow[fid] else sid
    phaseNeeded = (inid + 1) % 3 == vacid

    def su2pref(k):
        return 1. / np.sqrt(k + 1) * \
            (1. if k % 2 == 0 or not phaseNeeded else -1.)

    return {'fermionic': fpref, 'SU(2)': su2pref}


def _prefAdj(coupling, leg):
    """Prefactor for the creation of an adjoint

    Only implemented for regular three-legged tensors with an (in, in, out)
    flow and their adjoints at the moment.
    """
    if len(coupling) != 1:
        raise NotImplementedError("Only for three-legged tensors")

    flow = tuple(c[1] for c in coupling[0])
    if flow != (True, True, False) and flow != (True, False, False):
        raise NotImplementedError("Only (in, in, out) and its adjoint allowed")

    if flow == (True, False, False):
        def loop(x):
            return x[::-1]
    else:
        def loop(x):
            return x
    fid = [c[0] for c in loop(coupling[0])].index(leg) if leg else None

    fpref = {
        0: lambda key: 1. if loop(key[0])[1] % 2 == 0 else -1.,
        1: lambda key: 1. if loop(key[0])[0] % 2 == 0 else -1.,
        2: lambda key: 1.,
        None: lambda key: 1. if loop(key[0])[2] % 2 == 0 else -1.,
    }[fid]

    def su2pref(key):
        sign = (1, 1, -1)
        return 1. if sum(s * k for s, k in
                         zip(sign, loop(key[0]))) % 4 == 0 else -1.
    return {'fermionic': fpref, 'SU(2)': su2pref}


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

    for c, s in zip(zip(*coupling), symmetries):
        if constraint[s](c, flow) is False:
            return False
    return True


def vacuumIrrep(symmetries):
    """Returns the vacuum Irrep
    """
    return (0,) * len(symmetries)

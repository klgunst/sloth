from sloth.tensor import Tensor, Leg
import sloth.symmetries as sls
import pytest
import itertools
import numpy as np


def svclose(a, b, helpers):
    assert a['symmetries'] == b['symmetries']
    assert set(x for x in a if isinstance(x, tuple)) == \
        set(x for x in a if isinstance(x, tuple))
    for k, v in a.items():
        if isinstance(k, tuple):
            assert helpers.close_paddy(a[k], b[k])


@pytest.fixture()
def create_tensors(symmetr):
    """Create a pair of tensors whith one leg common with either
    PxU(1)xU(1) or PxU(1)xSU(2) symmetry
    """
    rng = np.random.default_rng()

    def make_random_l(nrirs, maxd):
        """Make legs with random dimensions
        """
        return Leg(), [(k, rng.integers(1, maxd, endpoint=True))
                       for k in itertools.product(*[range(x) for x in nrirs])]

    def make_random_t(symmetr, legs, flow):
        """Make random tensors
        """
        A = Tensor(symmetr, [(l[0], f) for l, f in zip(legs, flow)])
        for x in itertools.product(*[l[1] for l in legs]):
            key = (tuple(y[0] for y in x),)
            shape = tuple(y[1] for y in x)
            if sls.is_allowed_coupling(key[0], flow, symmetr) \
                    and 0 not in shape:
                A[key] = rng.uniform(low=-1.0, high=1.0, size=shape)
        return A

    mirrep = {
        'fermionic': 2,
        'SU(2)': 5,
        'U(1)': 5
    }
    legs = [make_random_l([mirrep[s] for s in symmetr], 5) for x in range(5)]
    flow = [True, True, False]
    # orthogonalized
    T, R = make_random_t(symmetr, legs[2:], flow).qr(legs[2][0])
    # Ortho center
    S = make_random_t(symmetr, legs[:3], flow) @ R
    S *= 1. / S.norm()
    return S, T


class TestTensors:
    def test_QR_OneSite(self, create_tensors):
        """Doing a QR decomposition on a random 1-site tensor
        """
        S, T = create_tensors
        for leg in S.indexes:
            Q, R = S.qr(leg)
            assert Q.is_ortho(Q.connections(R).pop())
            assert S.isclose(Q @ R)

    def test_QR_TwoSite(self, create_tensors):
        """Doing a QR decomposition on a random 2-site tensor
        """
        S, T = create_tensors
        S = S @ T
        for leg in S.indexes:
            Q, R = S.qr(leg)
            assert S.isclose(Q @ R)

    def test_SVD(self, create_tensors):
        """Doing a SVD on a random 2-site tensor
        """
        S, T = create_tensors
        A = S @ T
        assert len(A.internallegs) == 1
        U, s, V = A.svd(leg=A.internallegs[0])
        for X in (U, V):
            nleg = set(X.indexes).difference(A.indexes).pop()
            assert X.is_ortho(nleg)
        assert A.isclose(U @ s @ V)

    def test_SvalConsistent(self, create_tensors, helpers):
        A, B = create_tensors
        R = {leg: A.qr(leg)[1] for leg in A.indexes}
        svalQR = {leg: r.svd(compute_uv=False) for leg, r in R.items()}

        first_con = A.connections(B).pop()
        U = R[first_con] @ B
        second_con = [x for x in U.indexes if x not in B.indexes][0]
        for leg in U.indexes:
            sv = U.qr(leg)[1].svd(compute_uv=False)
            if leg == second_con:
                svclose(sv, svalQR[first_con], helpers)
            else:
                svalQR[leg] = sv

        # Now making two-site object
        C = A @ B
        U, S, V = C.svd(leg=C.internallegs[0])
        svclose(S, svalQR[first_con], helpers)

        for X in [U @ S, V @ S]:
            for leg in X.indexes:
                sv = X.qr(leg)[1].svd(compute_uv=False)
                if leg in svalQR:
                    svclose(sv, svalQR[leg], helpers)

    def test_Swap0Consistent(self, create_tensors, helpers):
        """Swap of legs of a three-legged tensor
        """
        A, _ = create_tensors

        # Singular values along loose edges
        svals = {leg: A.qr(leg)[1].svd(compute_uv=False) for leg in A.indexes}
        permutes = [[1, 2, 0], [2, 0, 1], [0, 2, 1], [2, 1, 0], [1, 0, 2]]

        def inv(p):
            return [p.index(i) for i in range(len(p))]

        for p in permutes:
            B = A.shallowcopy()._swap0(0, p)
            for leg in B.indexes:
                svclose(B.qr(leg)[1].svd(compute_uv=False),
                        svals[leg], helpers)
            assert not B.isclose(A)
            assert A.isclose(B._swap0(0, inv(p)))  # Double swap is the same

    def test_Swap1Consistent(self, create_tensors, helpers):
        A, B = create_tensors

        # Singular values along loose edges
        R = {leg: A.qr(leg)[1] for leg in A.indexes}
        svalQR = {leg: r.svd(compute_uv=False) for leg, r in R.items()}
        B2 = R[A.connections(B).pop()] @ B
        for leg in set(B2.indexes).intersection(B.indexes):
            svalQR[leg] = B2.qr(leg)[1].svd(compute_uv=False)

        C = A @ B
        assert len(C.coupling) == 2
        cids = (0, 1)
        ids = [[ii for ii, (c, _) in enumerate(cc) if c not in C.internallegs]
               for cc in C.coupling]

        for ii in itertools.product(*ids):
            C2 = C.shallowcopy()._swap1(cids, ii)
            U, S, V = C2.svd(leg=C2.internallegs[0])

            # Checking singular values are consistent
            for X in [U @ S, V @ S]:
                for leg in set(X.indexes).intersection(set(svalQR)):
                    svclose(X.qr(leg)[1].svd(compute_uv=False), svalQR[leg],
                            helpers)

            assert not C.isclose(C2)
            assert C.isclose(C2._swap1(cids, ii))  # Double swap is the same

    def test_adjoint(self, create_tensors, helpers):
        A, B = create_tensors

        # Singular values along loose edges
        for leg in A.indexes:
            B = A.adj(leg)
            C = B.adj(set(B.indexes).difference(A.indexes).pop())  # second adj
            to_subst = set(C.indexes).difference(A.indexes).pop()
            C._coupling = C.substitutelegs([to_subst], [leg])
            C._indexes = tuple(x if x != to_subst else leg for x in C._indexes)
            assert A.isclose(C)

import jax.numpy as np
from jax import jit

_SYMMETRIES = ['fermionic', 'U(1)', 'SU(2)', 'C1', 'Ci', 'C2', 'Cs', 'D2',
               'C2v', 'C2h', 'D2h', 'seniority']


class Leg:
    """Class for the legs of the tensors.

    Attrs:
        id: The index of the leg
        These two are only viable for SU(2) tensors.
            phase: Possible presence of the $(-1)^{j - m}$ phase on the leg.
            pref: Possible presence of the $[j]$ prefactor on the leg.
    """
    def __init__(self, phase=False, pref=False, vacuum=False, leg=None):
        if leg is None:
            self._phase = phase
            self._pref = pref
            self._vacuum = vacuum
        else:
            self._phase = leg.phase
            self._pref = leg.pref
            self._vacuum = leg.vacuum

    @property
    def phase(self):
        return self._phase

    @property
    def pref(self):
        return self._pref

    @property
    def vacuum(self):
        return self._vacuum

    def __repr__(self):
        return f"<{str(self.__class__)[8:-2]} object at {hex(id(self))}:" + \
            f"({self.__dict__})>"

    def same(self, leg):
        return self.phase == leg.phase and self.pref == leg.pref


class Tensor:
    """Class for the symmetry-invariant tensors.

    Attrs:
        coupling: tuple(((leg1, isin), (leg2, isin), (leg3, isin)), ...)
        symmetries: List of the symmetries in the tensor.
    """
    def __init__(self, symmetries, coupling=None):
        invalids = [i for i in symmetries if i not in _SYMMETRIES]
        if invalids:
            raise ValueError(f'Invalid symmetries inputted: {invalids}')

        symmetries = list(symmetries)
        symmetries.sort(reverse=True)
        self._symmetries = tuple(symmetries)
        self._coupling = None
        self.coupling = coupling
        self._data = {}

    @property
    def coupling(self):
        return self._coupling

    @property
    def indexes(self):
        return self._indexes

    @property
    def internallegs(self):
        return self._internallegs

    @coupling.setter
    def coupling(self, coupling):
        if coupling is None:
            self._coupling = None
            self._internallegs = None
            self._indexes = None
            return

        if not isinstance(coupling[0][0], (list, tuple)):
            coupling = (coupling,)

        if sum([len(c) == 3 for c in coupling]) != len(coupling):
            raise ValueError('coupling should be an (x, 3 * (Leg, bool)) or '
                             '(3 * (Leg, bool)) nested list/tuple.')

        for x in (el for c in coupling for el in c):
            if not (isinstance(x[0], Leg) and isinstance(x[1], bool)):
                raise ValueError('coupling should be an (x, 3 * (Leg, bool)) '
                                 'or (3 * (Leg, bool)) nested list/tuple.')

        FIRSTTIME = self._coupling is None
        self._coupling = tuple(tuple(tuple(el) for el in c) for c in coupling)

        # First time setting of the coupling
        if FIRSTTIME:
            self._indexes = []
            self._internallegs = []

            flat_c = tuple(el for c in self.coupling for el in c)
            flat_cnb = tuple(x for x, y in flat_c)
            uniques = set(flat_cnb)
            for x in flat_cnb:
                if x not in uniques:
                    continue

                uniques.remove(x)
                occurences = flat_cnb.count(x)
                if occurences == 1:
                    self._indexes.append(x)
                elif occurences == 2:
                    foc = flat_cnb.index(x)
                    soc = flat_cnb[foc + 1:].index(x) + foc + 1
                    if flat_c[foc][1] == flat_c[soc][1]:
                        raise ValueError('Internal bond is not in-out')

                    self._internallegs.append(x)
                else:
                    raise ValueError('Same leg occurs {occurences} times')

    def flowof(self, leg):
        if leg not in self.indexes:
            raise ValueError('Leg not an outer leg')
        x, y = self.coupling_id(leg)
        return self.coupling[x][y][1]

    @property
    def symmetries(self):
        return self._symmetries

    def __repr__(self):
        d = {k: v for k, v in self.__dict__.items() if k != '_data'}
        s = f"<{str(self.__class__)[8:-2]} object at {hex(id(self))}:({d})\n"
        return s + ''.join([f"{k}:\n{v}\n" for k, v in self.items()]) + '>'

    def __setitem__(self, index, value):
        self._data[index] = value

    def __getitem__(self, index):
        return self._data[index]

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return len(self._data)

    def items(self):
        return self._data.items()

    def ravel(self):
        """Puts the sparse tensor in a onedimensional array.
        """
        return np.concatenate([d.ravel() for _, d in self.items()])

    @property
    def size(self):
        return sum([d.size for _, d in self.items()])

    def coupling_id(self, leg):
        """Finds first occurence of leg in the coupling
        """
        for i, x in enumerate(self.coupling):
            for j, (y, _) in enumerate(x):
                if y == leg:
                    return i, j
        return None

    def connections(self, B):
        """Returns the connections between `Tensor` `self` and `B` and thus if
        they can be contracted along these connections.

        It is possible that the flows do not match and an extra vacuum-coupling
        should be inserted.
        """
        if not isinstance(B, Tensor):
            raise TypeError(f'{B} is not of {self.__class__}')
        return [x for x in self.indexes if x in B.indexes]

    def substitutelegs(self, old, new):
        """returns a coupling tuple where oldleg in self is swapped with
        newleg.
        """
        flatc = [el[0] for tpl in self.coupling for el in tpl]
        for ol in old:
            if ol not in flatc:
                raise ValueError(f'{ol} not a coupling for {self}')

        return tuple(
            tuple((new[old.index(x)], y) if x in old else (x, y) for x, y in z)
            for z in self.coupling)

    def __matmul__(self, B):
        """Trying to completely contract self and B for all matching bonds.
        """
        connections = self.connections(B)
        if not connections:
            raise ValueError(f'No connections found between {self} and {B}')

        return self.contract(B, (connections,) * 2).simplify()

    def contract(self, B, legs):
        """This function contract legs of two tensors.

        Args:
            legs: (2,) array_like of `Leg` objects.
        """
        if self.symmetries != B.symmetries:
            raise ValueError('Symmetries of the two tensors not same.')

        if len(legs[0]) != len(legs[1]):
            raise ValueError('legs should be (2,) array_like')

        AB = (self, B)
        if [len(l) for l in legs] != \
                [sum(lid in T.indexes for lid in l) for l, T in zip(legs, AB)]:
            raise ValueError('Not all given legs are loose for the tensors.')

        for l1, l2 in zip(legs[0], legs[1]):
            if self.flowof(l1) == B.flowof(l2):
                raise ValueError(
                    f'Contraction over {l1}, {l2} clashes for the flow,'
                    'I could fix this by automatically inserting a vacuum')

            if not l1.same(l2):
                raise ValueError(f'{l1} and {l2} do not have same prefactors '
                                 'on it. I could probably fix that myself')

        C = Tensor(self.symmetries)
        fcoup = tuple(T.indexes for T in AB)

        # index for the tensor itself of each leg
        oid = [[fl.index(el) for el in ll] for fl, ll in zip(fcoup, legs)]

        # coupling index for each leg
        cid = [[T.coupling_id(el) for el in ll] for T, ll in zip(AB, legs)]

        # equal amount new internal legs as pairs of legs
        ilegs = [Leg(leg=l) for l in legs[0]]

        # Coupling and thus keys for the dictionary are
        # [self.coupling, B.coupling] appended to each other with the
        # appropriate legs substituted.
        C.coupling = tuple(el for T, ll in zip(AB, legs)
                           for el in T.substitutelegs(ll, ilegs))
        for Ak, Abl in self.items():
            Akcid = [Ak[x][y] for x, y in cid[0]]
            for Bk, Bbl in B.items():
                if [Bk[x][y] for x, y in cid[1]] == Akcid:
                    C[(*Ak, *Bk)] = np.tensordot(Abl, Bbl, oid)

        index = [x for l, T in zip(legs, AB) for x in T.indexes if x not in l]

        # Same length
        assert len(index) == len(C._indexes)
        assert len(set(index)) == len(C._indexes)
        assert not [ii for ii in index if ii not in C._indexes]
        C._indexes = index

        return C

    def _manipulate_coupling(self, mappingf, prefactorf):
        """General function for manipulation of the coupling

        Args:
            mappingf: Function that maps the old keys to new keys of the tensor
            prefactorf: Function that gives the prefactor given the old key and
            new key for the transformation.
        """

        ndata = {}
        for okey in self:
            for nkey in mappingf(okey):
                prefactor = prefactorf(okey, nkey)

                if nkey in ndata:
                    ndata[nkey] += prefactor * self[okey]
                else:
                    ndata[nkey] = prefactor * self[okey]
        self._data = ndata

    def coupling_swap(self, cid, permute):
        """
        Permutes the indices in a given coupling.

        Args:
            cid: The coupling id in which to permute.
            permute: Permutation array.
        """
        permute = tuple(permute)
        if len(permute) != 3 or 0 not in permute or \
                1 not in permute or 2 not in permute:
            raise ValueError("Permutation array should be 0, 1, 2 shuffled")

        oc = self.coupling[cid]
        sc = tuple(oc[p] for p in permute)
        self.coupling = tuple(sc if i == cid else c for i, c in
                              enumerate(self.coupling))

        def mappingf(okey):
            yield tuple(
                tuple(okey[i][p] for p in permute) if i == cid else c
                for i, c in enumerate(okey)
            )

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

        fpref = {(0, 1, 2): fpref_123, (1, 0, 2): fpref_213,
                 (0, 2, 1): fpref_132, (2, 1, 0): fpref_321,
                 (2, 0, 1): fpref_312, (1, 2, 0): fpref_231}[permute]
        fids = tuple(ii for ii, ss in enumerate(self.symmetries)
                     if ss == 'fermionic')

        if 'SU(2)' in self.symmetries:
            raise NotImplementedError

        @jit
        def prefactorf(okey, nkey):
            # return 1. for empty array
            return np.prod([fpref(*[k[ii] for k in okey]) for ii in fids])

        self._manipulate_coupling(mappingf, prefactorf)

    def _remove_vacuumcoupling(self):
        """Removes couplings to the vacuum, if the tensor has multiple
        couplings.

        Only removes the vacuum if other two legs in the coupling have an
        opposite flow.

        Returns:
            If a vacuum has been removed.
        """
        if len(self.coupling) == 1 or \
                sum([ll.vacuum for ll in self.indexes]) == 0:
            return False

        flag = False
        for vac in self.indexes:
            if vac.vacuum:
                id0, id1 = self.coupling_id(vac)
                vac_coupling = self.coupling[id0]
                fid, sid = [[1, 2], [0, 2], [0, 1]][id1]
                fleg, sleg = vac_coupling[fid], vac_coupling[sid]

                # Other legs do not have the same flow
                if fleg[1] is not sleg[1]:
                    flag = True
                    break

        # No suitable vacuum found do remove.
        if not flag:
            return False

        def mappingf(okey):
            yield tuple(c for ii, c in enumerate(okey) if ii != id0)

        def fprefactor_1(a, b):
            return 1.

        def fprefactor_2(a, b):
            return -1. if a & 1 and b & 1 else 1.

        fpref = fprefactor_1 if fleg[1] else fprefactor_2
        fids = tuple(ii for ii, ss in enumerate(self.symmetries)
                     if ss == 'fermionic')

        @jit
        def prefactorf(okey, nkey):
            # return 1. for empty array
            return np.prod([fpref(*[k[ii] for k in okey]) for ii in fids])

        self._manipulate_coupling(mappingf, prefactorf)

        # fleg or sleg (one of the other legs in the coupling) is internal
        # Or both. remove either sleg or fleg and substitute it by the other
        # everywhere in the couplings
        if fleg[0] in self._internallegs:
            to_rm, to_swap = fleg[0], sleg[0]
        else:
            to_rm, to_swap = sleg[0], fleg[0]

        # Substitute the internal by the other leg
        temp = self.substitutelegs([to_rm], [to_swap])
        # Remove the coupling
        self._coupling = tuple(c for ii, c in enumerate(temp) if ii != id0)

        # Remove vacuum from index
        ii = self._indexes.index(vac)
        self._indexes.remove(vac)

        self._internallegs.remove(to_rm)

        for k, v in self.items():
            self[k] = np.squeeze(v, axis=ii)
        return True

    def simplify(self):
        """This function tries to simplify a tensor by changing it's coupling.

        Types of simplifications:
            * Removes couplings to the vacuum, if the tensor has multiple
              couplings.
        """
        # Iteratively remove vacuums until finished
        while self._remove_vacuumcoupling():
            pass

        return self

    def qr(self, leg):
        """Executes a QR decomposition for one of the legs.

        This leg can not be an internal leg, no prefactors from symmetry
        sectors needed.

        leg is ingoing:
            R will couple as (old leg, vacuum -> new leg)
            new leg is R -> Q
        leg is outgoing:
            R will couple as (new leg, vacuum -> old leg)
            new leg is Q -> R

        self is appropriately changed
        """
        if leg in self.internallegs:
            raise ValueError(f'{leg} is internal.')

        ingoing = self.flowof(leg)
        i1, i2 = self.coupling_id(leg)
        f_id = self.indexes.index(leg)

        R = Tensor(self.symmetries)
        Q = Tensor(self.symmetries)

        nleg = Leg(leg=leg)
        R.coupling = ((leg, True), (Leg(vacuum=True), True), (nleg, False)) if\
            ingoing else ((nleg, True), (Leg(vacuum=True), True), (leg, False))

        Q.coupling = self.substitutelegs([leg], [nleg])
        Q._indexes = tuple(i if i != leg else nleg for i in self._indexes)
        vacuum = (0,) * len(self.symmetries)

        assert [x == nleg for x in R.connections(Q)] == [True]

        keys = set([k[i1][i2] for k in self])
        transp = list(range(len(self.indexes)))
        transp.pop(f_id)
        transp.append(f_id)
        transp = np.array(transp)
        i_transp = np.argsort(transp)

        for key in keys:
            blocks = [k for k in self if k[i1][i2] == key]

            leading_dim = set(self[k].shape[f_id] for k in blocks)
            # check if dimension is consistent everywhere
            assert len(leading_dim) == 1
            leading_dim = leading_dim.pop()

            size = sum(self[k].size for k in blocks)
            assert size % leading_dim == 0

            # Moving all needed blocks into one matrix
            Aarr = [np.transpose(self[block], transp).reshape(-1, leading_dim)
                    for block in blocks]

            Aspl = np.cumsum(np.array([r.shape[0] for r in Aarr[:-1]]))

            q, r = np.linalg.qr(np.vstack(Aarr))
            r = np.expand_dims(r, axis=1)
            R[((key, vacuum, key),)] = r.T if ingoing else r

            # moving back all the blocks into the original tensor
            for block, x in zip(blocks, np.split(q, Aspl)):
                Q[block] = np.transpose(
                    x.reshape(np.array(self[block].shape)[transp]), i_transp)

        return Q, R

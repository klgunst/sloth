from sloth.symmetries import _SYMMETRIES, symswap
import jax.numpy as np


class Leg:
    """Class for the legs of the tensors.

    Attrs:
        id: The index of the leg
        These two are only viable for SU(2) tensors.
            SU2phase: Possible presence of the $(-1)^{j - m}$ phase on the leg.
            SU2pref: Possible presence of the $[j]$ prefactor on the leg.
    """
    def __init__(self, SU2phase=False, SU2pref=False, vacuum=False):
        self._SU2phase, self._SU2pref, self._vacuum = SU2phase, SU2pref, vacuum

    @property
    def SU2phase(self):
        return self._SU2phase

    @property
    def SU2pref(self):
        return self._SU2pref

    @property
    def vacuum(self):
        return self._vacuum

    def __repr__(self):
        return f"<{str(self.__class__)[8:-2]} object at {hex(id(self))}:" + \
            f"({self.__dict__})>"

    def same(self, leg):
        return self.SU2phase == leg.SU2phase and self.SU2pref == leg.SU2pref


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

        self._coupling = tuple(tuple(tuple(el) for el in c) for c in coupling)

        for x in (el for c in coupling for el in c):
            if not (isinstance(x[0], Leg) and isinstance(x[1], bool)):
                raise ValueError('coupling should be an (x, 3 * (Leg, bool)) '
                                 'or (3 * (Leg, bool)) nested list/tuple.')

        FIRSTTIME = self._coupling is None

        # First time setting of the coupling
        if FIRSTTIME:
            self._indexes = []
            self._internallegs = []

            flat_c = tuple(el[0] for c in self.coupling for el in c)
            flat_cnb = tuple(x for x, y in flat_c)
            for x in set(flat_cnb):
                occurences = flat_cnb.count(x)
                if occurences == 1:
                    self._indexes.append(x)
                elif occurences == 2:
                    foc = flat_cnb.index(x)
                    soc = flat_cnb[foc + 1:].index(x) + foc + 1
                    if flat_c[foc][1] == flat_c[soc][2]:
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

    @property
    def size(self):
        return sum([d.size for _, d in self._data.items()])

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
                raise ValueError('{l1} and {l2} do not have same prefactors on'
                                 ' it. I could probably fix that myself')

        C = Tensor(self.symmetries)
        fcoup = tuple(T.indexes for T in AB)

        # index for the tensor itself of each leg
        oid = [[fl.index(el) for el in ll] for fl, ll in zip(fcoup, legs)]

        # coupling index for each leg
        cid = [[T.coupling_id(el) for el in ll] for T, ll in zip(AB, legs)]

        # equal amount new internal legs as pairs of legs
        ilegs = [Leg(**l.__dict__) for l in legs[0]]

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

        indexes = [list(T.indexes) for T in AB]
        for ll, indx in zip(AB, indexes):
            for x in ll:
                indx.remove(x)
        # appended
        index = indexes[0].extend(indexes[1])
        # Same length
        assert len(index) == C._indexes
        assert len(set(index)) == len(C._indexes)
        assert not [ii for ii in index if ii not in C._indexes]

        C._indexes = index

        return C

    def coupling_swap(self, cid, permute):
        """
        """
        oc = self.coupling[cid]
        sc = tuple(oc[p] for p in permute)
        self.coupling = tuple(sc if i == cid else c for i, c in
                              enumerate(self.coupling))

        ndata = {}
        for key in self:
            ok = key[cid]
            nk = tuple(ok[p] for p in permute)
            nkey = tuple(nk if i == cid else c for i, c in enumerate(key))

            prefactor = np.prod([symswap(ss, [kk[i] for kk in ok], permute)
                                 for i, ss in enumerate(self.symmetries)])

            self[key] *= prefactor
            ndata[nkey] = self[key]
        self._data = ndata

    def simplify(self):
        """This function tries to simplify a tensor by changing it's coupling.

        Types of simplifications:
            * Removes couplings to the vacuum, if the tensor has multiple
              couplings.
        """
        # Remove couplings with vacuum leg in first or second thingy
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

        nleg = Leg(*leg.__dict__) if ingoing else Leg(*leg.__dict__)
        R.coupling = ((leg, True), (Leg(vacuum=True), True), (nleg, False)) if\
            ingoing else ((nleg, True), (Leg(vacuum=True), True), (leg, False))

        Q.coupling = self.substitutelegs([leg], [nleg])
        Q._indexes = tuple(i if i != leg else nleg for i in self._indexes)
        vacuum = (0,) * len(self.symmetries)

        assert [x == nleg for x in R.connections(self)] == [True]

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

            Aspl = np.cumsum([r.shape[0] for r in Aarr[:-1]])

            q, r = np.linalg.qr(np.vstack(Aarr))
            r = np.expand_dims(r, axis=1)
            R[((key, vacuum, key),)] = r.T if ingoing else r

            # moving back all the blocks into the original tensor
            for block, x in zip(blocks, np.split(q, Aspl)):
                Q[block] = np.transpose(
                    x.reshape(np.array(self[block].shape)[transp]), i_transp)

        return Q, R

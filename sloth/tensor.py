import networkx as nx
import math
import numpy as np
import itertools
import sloth.symmetries as sls

_SYMMETRIES = ['fermionic', 'U(1)', 'SU(2)', 'C1', 'Ci', 'C2', 'Cs', 'D2',
               'C2v', 'C2h', 'D2h', 'seniority']


class Leg:
    """Class for the legs of the tensors.

    Attrs:
        id: The index of the leg
        These two are only viable for SU(2) tensors.
    """
    def __init__(self, vacuum=False, leg=None):
        if leg is None:
            self._vacuum = vacuum
        else:
            self._vacuum = leg.vacuum

    @property
    def vacuum(self):
        return self._vacuum

    def __repr__(self):
        return f"<{str(self.__class__)[8:-2]} object at {hex(id(self))}:" + \
            f"({self.__dict__})>"


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

        # symmetries = list(symmetries)
        # symmetries.sort(reverse=True)
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
        if self._coupling is not None:
            raise AttributeError('Can\'t reinit a coupling')

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

    def getSymmIds(self, symm):
        return [i for i, s in enumerate(self.symmetries) if s == symm]

    @property
    def symmetries(self):
        return self._symmetries

    def isclose(self, A, **kwargs):
        """Compares two tensors.

        Args:
            A: The tensor to compare with
            kwargs: Arguments to pass to np.allclose

        Returns:
            True if the legs are the same, the symmetries are the same, the
            coupling is the same and the elements are close to each other.
        """
        # Symmetries not the same
        if self.symmetries != A.symmetries:
            return False

        # Indexes not necessarily the same order
        if set(self.indexes) != set(A.indexes):
            return False
        permute = tuple(self.indexes.index(x) for x in A.indexes)

        # Coupling should be the same
        if self.substitutelegs(self.internallegs, A.internallegs) \
                != A.coupling:
            return False
        return self.elclose(A, permute, **kwargs)

    def elclose(self, A, permute=None, **kwargs):
        """Compares the elements of two tensors
        """
        # List of unique keys
        keys = set(self).union(set(A))
        for key in keys:
            try:
                a = self[key]
                if permute:
                    a = np.transpose(a, axes=permute)
            except KeyError:
                a = 0.
            b = A._data.get(key, 0.)

            if not np.allclose(a, b, **kwargs):
                return False
        return True

    def get_legs(self):
        return [[x[0] for x in y] for y in self.coupling]

    def get_couplingnetwork(self):
        network = nx.MultiDiGraph()
        network.add_nodes_from(self.coupling)
        for ll in self.internallegs:
            incoupl = set(c for c in self.coupling if (ll, True) in c)
            outcoupl = set(c for c in self.coupling if (ll, False) in c)
            assert len(incoupl) == 1 and len(outcoupl) == 1
            network.add_edge(outcoupl.pop(), incoupl.pop(), leg=ll)
        return network

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

    def iterate(self, keys, couple_ids):
        """Iterations over the correct blocks which have value keys at the
        couple_ids.
        """
        # TODO: Better second iteration over only valid keys?
        for key, block in self.items():
            if [key[x][y] for x, y in couple_ids] == keys:
                yield key, block

    def items(self):
        return self._data.items()

    def ravel(self):
        """Puts the sparse tensor in a onedimensional array.
        """
        return np.concatenate([d.ravel() for _, d in self.items()])

    def norm(self):
        """Calculates the norm of the tensor
        """
        return np.linalg.norm(self.ravel())

    @property
    def size(self):
        return sum([d.size for _, d in self.items()])

    @property
    def shape(self):
        """The dense shape
        """
        # dictionaries with the stored dimensions for each key
        shapes = [{} for x in range(len(self.indexes))]
        cids = [self.coupling_id(leg) for leg in self.indexes]
        for k, v in self.items():
            for cid, shapedict, ii in zip(cids, shapes, range(len(cids))):
                key = k[cid[0]][cid[1]]
                if key in shapedict:
                    assert shapedict[key] == v.shape[ii]
                else:
                    shapedict[key] = v.shape[ii]
        return tuple(sum(x for _, x in d.items()) for d in shapes)

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
        return set(self.indexes).intersection(B.indexes)

    def substitutelegs(self, old, new):
        """returns a coupling tuple where oldleg in self is swapped with
        newleg.
        """
        # making dictionary
        if not hasattr(old, '__iter__'):
            old, new = [old], [new]
        dic = {o: n for o, n in zip(old, new)}
        return tuple(
            tuple((dic.get(x, x), y) for x, y in z) for z in self.coupling)

    def swaplegs(self, swapdict):
        self._coupling = tuple(tuple((swapdict.get(x, x), y) for x, y in z)
                               for z in self.coupling)
        self._indexes = tuple(swapdict.get(x, x) for x in self._indexes)
        self._internallegs = tuple(swapdict.get(x, x) for x in
                                   self._internallegs)
        return self

    def metacopy(self):
        """copy metadata
        """
        X = Tensor(self.symmetries, coupling=self.coupling)
        X._indexes = list(self.indexes)
        return X

    def shallowcopy(self):
        X = self.metacopy()
        X._data = self._data
        return X

    def __matmul__(self, B):
        """Trying to completely contract self and B for all matching bonds.
        """
        if isinstance(B, dict):
            X = self.metacopy()

            if B['leg'] not in X.indexes:
                raise ValueError('Leg of singular values not an indexes '
                                 'of self')

            if B['symmetries'] != X.symmetries:
                raise ValueError('Not same symmetries')

            x, y = X.coupling_id(B['leg'])
            for k in self:
                newshape = [1] * len(self[k].shape)
                newshape[X.indexes.index(B['leg'])] = -1
                X[k] = self[k] * B[k[x][y]].reshape(newshape)

            return X

        connections = self.connections(B)
        if not connections:
            raise ValueError(f'No connections found between {self} and {B}')

        return self.contract(B, (list(connections),) * 2).simplify()

    def __imatmul__(self, B):
        """Trying to completely contract self and B for all matching bonds.
        """
        if isinstance(B, dict):
            if B['leg'] not in self.indexes:
                raise ValueError('Leg of singular values not an indexes '
                                 'of self')

            if B['symmetries'] != self.symmetries:
                raise ValueError('Not same symmetries')

            x, y = self.coupling_id(B['leg'])
            for k in self:
                newshape = [1] * len(self[k].shape)
                newshape[self.indexes.index(B['leg'])] = -1
                self[k] *= B[k[x][y]].reshape(newshape)
            return self

        connections = self.connections(B)
        if not connections:
            raise ValueError(f'No connections found between {self} and {B}')

        return self.contract(B, (list(connections),) * 2).simplify()

    def __imul__(self, a):
        for key in self:
            self[key] *= a
        return self

    def __itruediv__(self, a):
        for key in self:
            self[key] /= a
        return self

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

        C = Tensor(self.symmetries)

        # index for the tensor itself of each leg
        oid = [[T.indexes.index(el) for el in ll] for T, ll in zip(AB, legs)]

        # coupling index for each leg
        cid = [[T.coupling_id(el) for el in ll] for T, ll in zip(AB, legs)]

        # equal amount new internal legs as pairs of legs
        ilegs = [Leg(leg=l) for l in legs[0]]

        # Coupling and thus keys for the dictionary are
        # [self.coupling, B.coupling] appended to each other with the
        # appropriate legs substituted.
        C.coupling = tuple(el for T, ll in zip(AB, legs)
                           for el in T.substitutelegs(ll, ilegs))

        Akeys, Bkeys = {}, {}
        for T, ci, dic in zip((self, B), cid, (Akeys, Bkeys)):
            for key in T:
                kk = tuple(key[x][y] for x, y in ci)
                try:
                    dic[kk].append(key)
                except KeyError:
                    dic[kk] = [key]

        for kk in set(Akeys).intersection(Bkeys):
            for Ak in Akeys[kk]:
                Abl = self[Ak]
                for Bk in Bkeys[kk]:
                    Bbl = B[Bk]
                    C[(*Ak, *Bk)] = np.tensordot(Abl, Bbl, oid)

        index = [x for l, T in zip(legs, AB) for x in T.indexes if x not in l]

        # Same array with unique elements
        assert set(index) == set(C._indexes) and len(index) == len(set(index))
        C._indexes = index

        return C

    def _manipulate_coupling(self, mappingf, prefactorf):
        """General function for manipulation of the coupling

        Args:
            mappingf: Function that maps the old keys to new keys of the
            tensor. Should be a generator of the different keys.
            prefactorf: Function that returns the needed prefactor given the
            old key and new key for the transformation.
        """

        ndata = {}
        for okey in self:
            for nkey in mappingf(okey):
                pref = prefactorf(okey, nkey)
                if math.isclose(pref, 0, rel_tol=0, abs_tol=1e-12):
                    continue

                b = np.multiply(pref, self[okey])
                if nkey in ndata:
                    ndata[nkey] += b
                else:
                    ndata[nkey] = b
        self._data = ndata

    def _swap0(self, cid, permute):
        """
        Permutes the indices in a given coupling.

        Args:
            cid: The coupling id in which to permute.
            permute: Permutation array.
        """
        from collections import Counter
        permute = tuple(permute)
        if Counter(permute) != Counter(range(3)):
            raise ValueError("Permutation array should be 0, 1, 2 shuffled")

        def permute_key(key):
            return tuple(tuple(key[cid][p] for p in permute) if i == cid else c
                         for i, c in enumerate(key))

        self._coupling = permute_key(self.coupling)

        def mappingf(okey):
            yield permute_key(okey)

        prefdict = sls._prefswap0(permute)

        def prefactorf(okey, nkey):
            kk = [x for x in zip(*okey[cid])]
            return np.prod([prefdict[ss](*k) if prefdict.get(ss, None)
                            else 1. for k, ss in zip(kk, self.symmetries)])

        self._manipulate_coupling(mappingf, prefactorf)
        return self

    def _swap1(self, cids, iids):
        """Switches two legs between two neighbouring couplings.
        """
        # The coupling indexes of the two legs to swap
        c1, c2 = cids
        # The index of the two legs to swap within the given coupling
        i1, i2 = iids
        assert c1 != c2

        # Get the connecting leg between the two couplings
        legs = self.get_legs()
        intersect = set(legs[c1]).intersection(set(legs[c2]))

        assert len(intersect) == 1  # Only one internal leg between couplings
        ileg = intersect.pop()
        # index of the internal leg in c1 and c2
        ii = [legs[cid].index(ileg) for cid in cids]

        assert ii[0] != i1 and ii[1] != i2
        # Check that the flow is consistent along the internal bond
        assert self.coupling[c1][ii[0]][1] is not self.coupling[c2][ii[1]][1]

        # Order such that first bond is in the one with out
        if self.coupling[c1][ii[0]][1]:
            c1, c2, i1, i2, ii = c2, c1, i2, i1, (ii[1], ii[0])
        assert not self.coupling[c1][ii[0]][1] and self.coupling[c2][ii[1]][1]

        def permute_key(key):
            copy = list(list(k) for k in key)
            copy[c1][i1], copy[c2][i2] = copy[c2][i2], copy[c1][i1]
            return copy
        self._coupling = tuple(tuple(c) for c in permute_key(self.coupling))
        f1, f2 = ([x[1] for x in self.coupling[c]] for c in (c1, c2))

        def mappingf(okey):
            nk = permute_key(okey)
            # All good interal symmetry sectors in for the swapped 1st coupling
            for k in sls.allowed_couplings(nk[c1], f1, ii[0], self.symmetries):
                # Assign the key of the internal leg
                nk[c1][ii[0]], nk[c2][ii[1]] = k, k
                if sls.is_allowed_coupling(nk[c2], f2, self.symmetries):
                    yield tuple(tuple(e) for e in nk)

        prefdict = sls._prefswap1((i1, i2), ii)

        def prefactorf(okey, nkey):
            return np.prod([prefdict.get(ss, lambda x, y: 1.)(
                [el[i] for j in (c1, c2) for el in okey[j]],
                [el[i] for j in (c1, c2) for el in nkey[j]]
            ) for i, ss in enumerate(self.symmetries)])

        self._manipulate_coupling(mappingf, prefactorf)
        return self

    def _swap2(self, cids, iids):
        """Switches two legs between two next-to-nearest neighbouring
        couplings.
        """
        # The coupling indexes of the two legs to swap
        c1, c2 = cids
        # The index of the two legs to swap within the given coupling
        i1, i2 = iids
        assert c1 != c2

        # Get the connecting coupling between the two couplings
        cnx = self.get_couplingnetwork().to_undirected(as_view=True)
        ci = set(nx.common_neighbors(cnx, *[self.coupling[ii] for ii in cids]))
        if len(ci) != 1:
            raise ValueError(f'cids: {cids} have {len(ci)} common neighbors')
        ci = self.coupling.index(ci.pop())

        # internal legs
        l1 = cnx.edges[self.coupling[c1], self.coupling[ci], 0]['leg']
        l2 = cnx.edges[self.coupling[c2], self.coupling[ci], 0]['leg']

        # index of the internal leg in c1 and c2
        legs = self.get_legs()
        il1, il2 = [[legs[x].index(ll) for x in (y, ci)]
                    for y, ll in zip(cids, (l1, l2))]

        assert il1[0] != i1 and il2[0] != i2
        assert il1[1] != il2[1]
        # Check that the flow is consistent along the internal bond
        assert self.coupling[c1][il1[0]][1] is not self.coupling[ci][il1[1]][1]
        assert self.coupling[c2][il2[0]][1] is not self.coupling[ci][il2[1]][1]

        def permute_key(key):
            copy = list(list(k) for k in key)
            copy[c1][i1], copy[c2][i2] = copy[c2][i2], copy[c1][i1]
            return copy
        f1, f2, fi = ([x[1] for x in self.coupling[c]] for c in (c1, c2, ci))
        self._coupling = tuple(tuple(c) for c in permute_key(self.coupling))

        # All good interal symmetry sectors in for the swapped 1st coupling
        nkeys = set(tuple(tuple(e) for e in permute_key(k)) for k in self)
        c1set = {}
        r11, r12 = set(range(3)).difference([il1[0]])
        for k in set(key[c1] for key in nkeys):
            kn = (k[r11], k[r12])
            if kn not in c1set:
                c1set[kn] = set(
                    sls.allowed_couplings(k, f1, il1[0], self.symmetries))
        c2set = {}
        r21, r22 = set(range(3)).difference([il2[0]])
        for k in set(key[c2] for key in nkeys):
            kn = (k[r21], k[r22])
            if kn not in c2set:
                c2set[kn] = set(
                    sls.allowed_couplings(k, f2, il2[0], self.symmetries))

        vac = sls.vacuumIrrep(self.symmetries)
        Z1 = set().union(*c1set.values())
        Z2 = set().union(*c2set.values())
        rf = set(range(3)).difference([il1[1], il2[1]]).pop()
        fit = [fi[rf], fi[il1[1]], fi[il2[1]]]
        oks = {(k1, k2): set(sls.allowed_couplings((vac, k1, k2),
                                                   fit, 0, self.symmetries))
               for k1, k2 in itertools.product(Z1, Z2)}

        def mappingf(okey):
            nk = permute_key(okey)
            set1 = c1set[(nk[c1][r11], nk[c1][r12])]
            set2 = c2set[(nk[c2][r21], nk[c2][r22])]
            for kk1 in set1:
                for kk2 in set2:
                    if nk[ci][rf] not in oks[(kk1, kk2)]:
                        continue

                    # Assign the key of the internal leg
                    nk[c1][il1[0]], nk[ci][il1[1]] = kk1, kk1
                    nk[c2][il2[0]], nk[ci][il2[1]] = kk2, kk2
                    yield tuple(tuple(e) for e in nk)

        prefdict = sls._prefswap2(iids, il1, il2, f1, f2, fi)

        def prefactorf(okey, nkey):
            flokey = [list(x) for x in
                      zip(*[el for j in (c1, c2, ci) for el in okey[j]])]
            flnkey = [list(x) for x in
                      zip(*[el for j in (c1, c2, ci) for el in nkey[j]])]
            return np.prod([prefdict.get(ss, lambda x, y: 1.)(o, n) for
                            o, n, ss in zip(flokey, flnkey, self.symmetries)])

        self._manipulate_coupling(mappingf, prefactorf)
        return self

    def _remove_vacuumcoupling(self):
        """Removes couplings to the vacuum, if the tensor has multiple
        couplings.

        Only removes the vacuum if other two legs in the coupling have an
        opposite flow.

        Returns:
            If a vacuum has been removed.
        """
        if len(self.coupling) == 1 or \
                [ll.vacuum for ll in self.indexes].count(True) == 0:
            return False

        flag = False
        for vac in self.indexes:
            if vac.vacuum:
                id0, id1 = self.coupling_id(vac)
                vac_coupling = self.coupling[id0]
                fid, sid = [x for x in range(3) if x != id1]
                fleg, sleg = vac_coupling[fid], vac_coupling[sid]

                # Other legs do not have the same flow
                if fleg[1] is not sleg[1]:
                    flag = True
                    break

        # No suitable vacuum found to remove.
        if not flag:
            return False

        def mappingf(okey):
            assert okey[id0][fid] == okey[id0][sid]
            yield tuple(c for ii, c in enumerate(okey) if ii != id0)

        prefdict = sls._prefremovevac(id1, [x[1] for x in vac_coupling])

        def prefactorf(okey, nkey):
            # return 1. for empty array
            return np.prod([prefdict.get(ss, lambda x: 1.)(okey[id0][fid][ii])
                            for ii, ss in enumerate(self.symmetries)])

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

    def _detectSimpleLoop(self):
        """Detects simple loops in the coupling of the tensor, i.e. "-<>- → ⊥"

        Returns:
            a dictionary with properties of the detected simple loop or
            None if no simple loops are detected
        """
        # Internal legs in each coupling
        incoupl = [set(x[0] for x in y if x[0] in self.internallegs)
                   for y in self.coupling]
        # The couplings which have a crossection of two
        cross = [(i, j) for i, x in enumerate(incoupl)
                 for j, y in enumerate(incoupl[i + 1:], start=i + 1)
                 if len(x.intersection(y)) == 2]

        # No Simple loops
        if len(cross) == 0:
            return None

        # Pick the first simple loop found
        loop = {'coupl': cross[0]}
        # The looping legs
        loop['lLegs'] = tuple(
            incoupl[loop['coupl'][0]].intersection(incoupl[loop['coupl'][1]]))

        # The loop indexes
        loop['lid'] = tuple(
            tuple([x[0] for x in self.coupling[c]].index(ll)
                  for c in loop['coupl']) for ll in loop['lLegs'])

        # The free indexes
        loop['fid'] = tuple(set(range(3)).difference(set(x)).pop()
                            for x in zip(*loop['lid']))

        # Flow not consistent for lid
        for x, y in loop['lid']:
            if self.coupling[loop['coupl'][0]][x][1] == \
                    self.coupling[loop['coupl'][1]][y][1]:
                return None

        # Flow not consistent for fid
        if self.coupling[loop['coupl'][0]][loop['fid'][0]][1] == \
                self.coupling[loop['coupl'][1]][loop['fid'][1]][1]:
            return None

        # Re-sort such that the first coupling is ingoing
        if not self.coupling[loop['coupl'][0]][loop['fid'][0]][1]:
            loop['coupl'] = (loop['coupl'][1], loop['coupl'][0])
            loop['lid'] = tuple((x[1], x[0]) for x in loop['lid'])
            loop['fid'] = (loop['fid'][1], loop['fid'][0])

        assert self.coupling[loop['coupl'][0]][loop['fid'][0]][1]

        return loop

    def _removeSimpleLoop(self):
        """Removes a simple loop in the coupling, i.e. "-<>- → ⊥". The new
        introduced leg is a vacuum.

        Returns:
            True if a loop is removed, False otherwise
        """
        loop = self._detectSimpleLoop()
        if loop is None:
            return False
        vackey = sls.vacuumIrrep(self.symmetries)
        vacleg = Leg(vacuum=True)

        prefdict = sls._prefremoveSimpleLoop(loop, self.coupling)

        def keyMapping(newkey, oldkeys):
            return tuple([tuple(newkey)] + [c for ii, c in enumerate(oldkeys)
                                            if ii not in loop['coupl']])

        # Extracts the keys for the in and out legs of the simple loop
        def extractInAndOutKeys(key):
            return [key[x][y] for x, y in zip(loop['coupl'], loop['fid'])]

        self._coupling = keyMapping(
            ((vacleg, True), *extractInAndOutKeys(self.coupling)),
            self.coupling)

        def mappingf(okey):
            # The symmetry in the inkey (ik) and outkey (ok) should be the same
            ik, ok = extractInAndOutKeys(okey)
            if ik == ok:
                yield keyMapping((vackey, ik, ok), okey)

        # Remove internallegs belonging to the loop
        self._internallegs.remove(loop['lLegs'][0])
        self._internallegs.remove(loop['lLegs'][1])

        def prefactorf(okey, nkey):
            # assume that everything is consistent!
            keysw = tuple(x for x in zip(*okey[loop['coupl'][0]],
                                         *okey[loop['coupl'][1]]))
            return np.prod([prefdict.get(ss, lambda x: 1.)(k)
                            for k, ss in zip(keysw, self.symmetries)])

        self._manipulate_coupling(mappingf, prefactorf)

        # Add vacuum from index (last index)
        self._indexes.append(vacleg)
        for k in self:
            self[k] = np.expand_dims(self[k], self[k].ndim)

        return True

    def _detectTripleLoop(self):
        # Couplings that are connected to each other
        legs = self.get_legs()
        nc = [set(i for i, x in enumerate(legs) if x != c
                  and set(x).intersection(c)) for c in legs]
        for ii, ni in enumerate(nc):
            for jj in ni:
                s = ni.intersection(nc[jj])
                if s:
                    # There is a loop
                    kk = s.pop()
                    assert ii in nc[kk] and jj in nc[kk] and ii in nc[jj]

                    il, jl, kl = [[k for k, _ in self.coupling[x]]
                                  for x in (ii, jj, kk)]
                    # flow from ii to jj
                    iitojj = [y for x, y in self.coupling[ii] if x in jl][0]
                    # flow from kk to free
                    kktofree = [y for x, y in self.coupling[kk] if x not in jl
                                and x not in il][0]
                    if iitojj is kktofree:
                        il, jl = jl, il
                        ii, jj = jj, ii

                    # leg between jj and kk, index of jj
                    cid2 = set([x for x, y in enumerate(jl) if y in kl]).pop()
                    # free leg of ii, index of ii
                    cid1 = set([x for x, y in enumerate(il) if y not in kl
                                and y not in jl]).pop()
                    return (ii, jj), (cid1, cid2)

        return None

    def _removeTripleLoop(self):
        """Removes a triple loop.
        """
        loop = self._detectTripleLoop()
        if not loop:
            return False

        self._swap1(*loop)
        return True

    def simplify(self):
        """This function tries to simplify a tensor by changing it's coupling.

        Types of simplifications:
            * Removes any simple loops and changes them to couplings to the
              vacuum.
            * Removes couplings to the vacuum, if the tensor has multiple
              couplings.
        """
        # First remove all simple loops
        while self._removeTripleLoop():
            pass

        # First remove all simple loops
        while self._removeSimpleLoop():
            pass

        # Iteratively remove vacuums until finished
        while self._remove_vacuumcoupling():
            pass

        return self

    def qr(self, leg):
        """Executes a QR decomposition for one of the legs.

        This leg can not be an internal leg.

        leg is ingoing:
            R will couple as (vacuum, old leg -> new leg)
            new leg is R -> Q
        leg is outgoing:
            R will couple as (vacuum, new leg -> old leg)
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

        nleg = Leg()
        R.coupling = ((Leg(vacuum=True), True), (leg, True), (nleg, False)) if\
            ingoing else ((Leg(vacuum=True), True), (nleg, True), (leg, False))

        Q.coupling = self.substitutelegs([leg], [nleg])
        Q._indexes = tuple(i if i != leg else nleg for i in self.indexes)
        vacuum = (0,) * len(self.symmetries)

        assert R.connections(Q) == set([nleg])

        keys = set([k[i1][i2] for k in self])
        transp = list(range(len(self.indexes)))
        transp.pop(f_id)
        transp.append(f_id)
        transp = np.array(transp)
        i_transp = np.argsort(transp)

        SU2_ids = self.getSymmIds('SU(2)')
        internals = [
            (ii, jj) for ii, c in enumerate(self.coupling)
            for jj, (l, f) in enumerate(c) if l in self.internallegs and f]

        # For scaling of the internal legs
        def ipref(key):
            if SU2_ids:
                return np.prod([np.sqrt(key[x][y][i] + 1)
                                for (x, y) in internals for i in SU2_ids])
            else:
                return 1.

        for key in keys:
            lpref = np.prod([np.sqrt(key[i] + 1) for i in SU2_ids])
            blocks = [(k, ipref(k)) for k in self if k[i1][i2] == key]

            ldim = set(self[k].shape[f_id] for k, _ in blocks)
            # check if dimension is consistent everywhere
            assert len(ldim) == 1
            ldim = ldim.pop()

            size = sum(self[k].size for k, _ in blocks)
            assert size % ldim == 0

            # Moving all needed blocks into one matrix
            Aarr = [np.transpose(self[block], transp).reshape(-1, ldim) / pref
                    for block, pref in blocks]

            Aspl = np.cumsum(np.array([r.shape[0] for r in Aarr[:-1]]))

            q, r = np.linalg.qr(np.vstack(Aarr))
            newlead = q.shape[-1]
            thiskey = ((vacuum, key, key),)
            R[thiskey] = np.expand_dims((r.T if ingoing else r), axis=0)

            # moving back all the blocks into the original tensor
            for (block, pref), x in zip(blocks, np.split(q, Aspl)):
                new_shape = [self[block].shape[x] for x in transp]
                assert new_shape[-1] == ldim
                new_shape[-1] = newlead
                Q[block] = pref * lpref * \
                    np.transpose(x.reshape(new_shape), i_transp)

        return Q, R

    @staticmethod
    def truncate_svd(U, S, V, maxD):
        su2ids = [i for i, s in enumerate(S['symmetries']) if s == 'SU(2)']

        def multplic(key):
            return np.prod([key[i] + 1 for i in su2ids])

        wS = {k: multplic(k) * (v ** 2) for k, v in S.items()
              if isinstance(k, tuple)}
        sort = np.concatenate([v for k, v in wS.items()])

        if len(sort) <= maxD or maxD == 0:
            return U, S, V

        Smin = np.sort(sort)[::-1][maxD]
        scale = 1. / sum(np.sort(sort)[::-1][:maxD])
        for k in list(S):
            if isinstance(k, tuple):
                S[k] = S[k][wS[k] > Smin] * scale
                if len(S[k]) == 0:
                    del S[k]

        ci1, ci2 = U.coupling_id(S['leg'])
        U._data = {k: v[..., wS[k[ci1][ci2]] > Smin]
                   for k, v in U.items() if k[ci1][ci2] in S}

        ci1, ci2 = V.coupling_id(S['leg'])
        V._data = {k: v[wS[k[ci1][ci2]] > Smin, ...]
                   for k, v in V.items() if k[ci1][ci2] in S}

        return U, S, V

    def svd(self, leg=None, compute_uv=True, maxD=0):
        """Executes a SVD along the given leg(s).

        Args:
            leg: If None, it means that a vacuum-coupled tensor with only one
            coupling is inputted wich will be approriately decomposed.

            If an internal leg is given, the tensor is divided along this leg.

            compute_uv: False if only the singular values are needed.
        """
        # TODO: Can I do this shit cleaner?
        from networkx.algorithms.components import is_connected, \
            number_connected_components, connected_components

        SUid = self.getSymmIds('SU(2)')
        if leg:
            if leg not in self.internallegs:
                raise ValueError('Leg is not an internal one')

            U = Tensor(self.symmetries)
            V = Tensor(self.symmetries)
            S = {'symmetries': self.symmetries, 'leg': leg}

            lcid = self.coupling_id(leg)
            Skeys = set(key[lcid[0]][lcid[1]] for key in self)

            netw = self.get_couplingnetwork()
            assert is_connected(netw.to_undirected())
            for u, v, ll in netw.edges(data='leg'):
                if leg == ll:
                    netw.remove_edge(u, v)
                    break
            assert number_connected_components(netw.to_undirected()) == 2
            coupls = [tuple(c for c in G)
                      for G in connected_components(netw.to_undirected())]
            Uid = [True in [(leg, False) in c for c in coupl]
                   for coupl in coupls].index(True)
            Vid = [True in [(leg, True) in c for c in coupl]
                   for coupl in coupls].index(True)
            assert Uid != Vid

            U.coupling = coupls[Uid]
            V.coupling = coupls[Vid]

            assert leg in U.indexes and leg in V.indexes
            U._indexes.remove(leg)
            U._indexes.append(leg)
            V._indexes.remove(leg)
            V._indexes.insert(0, leg)

            permindexes = U.indexes[:-1] + V.indexes[1:]
            assert set(permindexes) == set(self.indexes)
            transp = np.array([self.indexes.index(ll) for ll in permindexes])
            Uids = transp[:len(U.indexes) - 1]
            Vids = transp[len(U.indexes) - 1:]
            Umap = [self.coupling.index(c) for c in U.coupling]
            Vmap = [self.coupling.index(c) for c in V.coupling]

            iU = [self.coupling_id(c) for c in U.internallegs]
            iV = [self.coupling_id(c) for c in V.internallegs]

            def pref(key, mp):
                return np.prod(
                    [np.sqrt(key[x][y][ii] + 1) for ii in SUid for x, y in mp])

            blockdict = {k: {} for k in Skeys}
            for k, b in self.items():
                blockdict[k[lcid[0]][lcid[1]]][k] = \
                    (b, pref(k, iU), pref(k, iV))

            for Skey in Skeys:
                dict_part = blockdict[Skey]
                Sprf = np.prod([np.sqrt(Skey[ii] + 1) for ii in SUid])

                Uslice, Ucur = {}, 0
                Vslice, Vcur = {}, 0
                for k, (b, Up, Vp) in dict_part.items():
                    Ukey = tuple([k[i] for i in Umap])
                    Vkey = tuple([k[i] for i in Vmap])

                    if Ukey not in Uslice:
                        Udims = [b.shape[ii] for ii in Uids]
                        Ud = np.prod(Udims)
                        Uslice[Ukey] = slice(Ucur, Ucur + Ud), Udims, Up
                        Ucur += Ud
                    if Vkey not in Vslice:
                        Vdims = [b.shape[ii] for ii in Vids]
                        Vd = np.prod(Vdims)
                        Vslice[Vkey] = slice(Vcur, Vcur + Vd), Vdims, Vp
                        Vcur += Vd

                memory = np.zeros((Ucur, Vcur))

                for k, (b, Up, Vp) in dict_part.items():
                    Ukey = tuple([k[i] for i in Umap])
                    Vkey = tuple([k[i] for i in Vmap])
                    uslice, _, _ = Uslice[Ukey]
                    vslice, _, _ = Vslice[Vkey]
                    ud = uslice.stop - uslice.start
                    vd = vslice.stop - vslice.start
                    memory[uslice, vslice] = \
                        np.transpose(b, transp).reshape(ud, vd) / (Up * Vp)

                # Finally do SVD
                if compute_uv:
                    u, s, v = np.linalg.svd(memory, full_matrices=False)
                else:
                    s = np.linalg.svd(memory, compute_uv=False)
                    S[Skey] = s / Sprf / Sprf
                    continue

                S[Skey] = s / Sprf / Sprf
                for key, (sl, dims, Up) in Uslice.items():
                    U[key] = u[sl, :].reshape(*dims, -1) * Up * Sprf
                for key, (sl, dims, Vp) in Vslice.items():
                    V[key] = v[:, sl].reshape(-1, *dims) * Vp * Sprf

            if compute_uv:
                return self.truncate_svd(U, S, V, maxD)
            else:
                return S
        else:
            # Plain svd of a R matrix. Only calculates the singular values
            if len(self.coupling) != 1:
                raise ValueError(
                    'For SVD with no leg specified, the tensor should be a '
                    'simple one with only 1 coupling to the vacuum.')
            try:
                Sid = [l.vacuum for l in self.indexes].index(True)
                _, Scid = self.coupling_id(self.indexes[Sid])
            except ValueError:
                raise ValueError(
                    'For SVD with no leg specified, the tensor should be a '
                    'simple one with only 1 coupling to the vacuum.')

            if compute_uv:
                raise ValueError('For SVD with no leg only allowed for '
                                 'calculating the singular values themselves.')

            S = {'symmetries': self.symmetries}
            Ucid = 0 if Scid != 0 else 1
            Vcid = 2 if Scid != 2 else 1

            def prefact(key):
                return np.prod([np.sqrt(k[ii] + 1) for ii in SUid])

            for key, block in self.items():
                k = key[0][Ucid]
                assert k == key[0][Vcid]
                S[k] = np.linalg.svd(np.squeeze(block, axis=Sid) / prefact(k),
                                     compute_uv=False)
            return S

    def adj(self, leg):
        """Returns an adjoint of the given tensor which is assumed to be
        orthogonalized with respect to the given leg.


        If leg is None than this specifies it is the orthogonality center.
        """
        adj = self.metacopy()

        # Substitute the loose leg by a new leg, if not ortho center
        nleg = Leg(leg=leg) if leg else None
        # Assign new internal legs
        ilegs = [Leg(leg=ll) for ll in self.internallegs]
        scoup = self.substitutelegs(self.internallegs + [leg]
                                    if self.internallegs else [leg],
                                    ilegs + [nleg])
        # Change the coupling flows appropriately
        adj._coupling = tuple(tuple((c[0], not c[1]) for c in reversed(x))
                              for x in scoup)
        adj._indexes = tuple(x if x != leg else nleg for x in adj._indexes)

        prefdict = sls._prefAdj(self.coupling, leg)

        def prefactorf(key):
            return np.prod([prefdict.get(ss, lambda x: 1.)(
                tuple(tuple(x[ii] for x in y) for y in key)
            ) for ii, ss in enumerate(self.symmetries)])

        adj._data = {}
        for key in self:
            adj[tuple(tuple(x for x in reversed(k)) for k in key)] = \
                self[key] * prefactorf(key)
        return adj

    def is_unity(self, **kwargs):
        """Checks if a tensor is the unit tensor.
        """
        # Only one coupling allowed
        if len(self.coupling) != 1:
            return False

        # Should contain one coupling with the vacuum
        if [l.vacuum for l, _ in self.coupling[0]].count(True) != 1:
            return False

        # The vacuum index
        vid = [l.vacuum for l in self.indexes].index(True)
        vid2 = [l.vacuum for l, _ in self.coupling[0]].index(True)
        oid1, oid2 = [ii for ii in range(3) if ii != vid2]
        id1, id2 = [ii for ii in range(3) if ii != vid]
        flow = [f for _, f in self.coupling[0]]
        prefdict = sls._prefremovevac(vid2, flow)

        def scalingfactor(k):
            return np.prod([prefdict.get(ss, lambda x: 1.)(kk)
                            for kk, ss in zip(k[oid1], self.symmetries)])

        for k, v in self.items():
            k = k[0]  # select key of first coupling
            assert k[oid1] == k[oid2]  # Should not occur when coupling to vac

            # Not square
            # if v.shape[id1] != v.shape[id2]:
            #     return False

            if not np.allclose(np.eye(v.shape[id1]), scalingfactor(k)
                               * np.squeeze(v, axis=vid), **kwargs):
                return False
        return True

    def is_ortho(self, leg, **kwargs):
        """Checks if a tensor is indeed an orthogonal tensor.

        Args:
            leg: The leg that should be kept loose during the contraction with
            it's adjoint.
        Returns:
            True if indeed an orthogonal tensor, otherwise False.
        """
        B = self @ self.adj(leg)  # Make and contract with the adjoint
        return B.is_unity(**kwargs)

    def swap(self, fl, sl):
        cids = [self.coupling_id(ll) for ll in (fl, sl)]
        cnx = self.get_couplingnetwork().to_undirected(as_view=True)
        if cids[0][0] == cids[1][0]:
            # Do _swap0
            permute = list(range(3))
            permute[cids[0][1]], permute[cids[1][1]] = \
                permute[cids[1][1]], permute[cids[0][1]]
            return self._swap0(cids[0][0], permute)
        elif self.coupling[cids[1][0]] in \
                cnx.neighbors(self.coupling[cids[0][0]]):
            return self._swap1(*[x for x in zip(*cids)])
        elif set(nx.common_neighbors(cnx,
                                     *[self.coupling[i] for i, _ in cids])):
            return self._swap2(*[x for x in zip(*cids)])
        else:
            raise NotImplementedError

    def swapCouplingTuples(self, permutation):
        """Swaps whole tuples of the coupling according to the permutation.
        """
        from collections import Counter

        if Counter(permutation) != Counter(range(len(self.coupling))):
            raise ValueError(f'Permutation {permutation} is not valid')

        def permute_key(key):
            return tuple(key[p] for p in permutation)

        self._coupling = permute_key(self.coupling)

        def mappingf(okey):
            yield permute_key(okey)

        self._manipulate_coupling(mappingf, lambda x, y: 1)
        return self

    def couplingAddapt(self, ncoupling):
        """Addapts the coupling of self to ncoupling if possible,
        else raises a ValueErorr.
        """
        from collections import Counter
        if Counter(y for x in self.coupling for y in x) \
                != Counter(y for x in ncoupling for y in x):
            raise ValueError('Couplings are not mutable into each other')

        # Reorder such the internal legs belong to corresp couplings
        permutation = [set([(l, f) in c for c in self.coupling].index(True)
                           for l, f in C if l in self.internallegs)
                       for C in ncoupling]

        if [len(x) for x in permutation] != [True] * len(permutation):
            raise ValueError('Couplings are not mutable into each other')
        self.swapCouplingTuples([x.pop() for x in permutation])

        # First set all the free legs on the right couplings
        ncl = tuple(tuple(x for x, y in c) for c in ncoupling)
        nl = self.get_legs()
        swapping = {l: (
            [l in c for c in ncl].index(True), [l in c for c in nl].index(True)
        ) for l in self.indexes}
        # Remove the legs that do not need a swapping
        swapping = {l: v for l, v in swapping.items() if v[0] != v[1]}

        # while swapping is not empty
        while swapping:
            assert len(swapping) % 2 == 0
            l1, swap = swapping.popitem()
            l2 = None
            for k, v in swapping.items():
                if v[0] == swap[1]:
                    # This is a good leg to swap with
                    l2 = k
                if v[1] == swap[0]:
                    # This is a very good leg to swap with
                    break
            assert l2 is not None
            self.swap(l1, l2)
            del swapping[l2]
        assert len(swapping) == 0

        # All legs are assigned to the right coupling at this point
        # Permute every coupling!
        for ii, (c1, c2) in enumerate(zip(self.coupling, ncoupling)):
            assert Counter(c1) == Counter(c2)  # They are equivalent
            self._swap0(ii, [c1.index(x) for x in c2])

        assert [y for x in self.coupling for y in x] \
            == [y for x in ncoupling for y in x]
        return self

    def optimizeBasis(self, legs, alpha=1., init=0., tol=1e-6):
        """Optimizes the two-site tensor with respect to two given physical
        legs.
        """
        from scipy.optimize import minimize
        from sloth.utils import renyi_entropy

        def entanglement(theta):
            from sloth.utils import flatten_svals
            A = self.rotate(theta[0], legs)
            S = A.svd(leg=A.internallegs[0], compute_uv=False)
            entropy = renyi_entropy(S, alpha)
            assert np.isclose(np.linalg.norm(flatten_svals(S)), 1.)
            return entropy

        res = minimize(entanglement, np.array(init), tol=tol,
                       bounds=[(-1.6, 1.6)])
        return res

    def rotate(self, theta, legs):
        """Rotates the physical legs (does not check if it is really a physical
        leg or not.
        """
        U, onew = rotationTensor(theta, self.symmetries, legs)
        B = U @ self
        new = list(onew)
        old = list(legs)
        if B.internallegs != self.internallegs:
            old.append(self.internallegs[0])
            new.append(B.internallegs[0])
        B.swaplegs({n: o for o, n in zip(old, new)})
        return B.couplingAddapt(self.coupling)


def rotationTensor(theta, symmetries, pl):
    from math import cos, sin

    if tuple(symmetries) != ('fermionic', 'U(1)', 'U(1)') and \
            tuple(symmetries) != ('fermionic', 'U(1)', 'SU(2)'):
        raise NotImplementedError(
            f'rotation Tensor not implemented for {symmetries}')

    ct, st = cos(theta), sin(theta)
    npl = [Leg(leg=l) for l in pl]
    il = Leg()
    U = Tensor(symmetries,
               coupling=(((npl[0], True), (npl[1], True), (il, False)),
                         ((il, True), (pl[1], False), (pl[0], False))))

    def valuesU1():
        # Occupations of one spatial orbital in U(1) x U(1)
        e, u, d, f = (0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)
        u2, d2, f2 = (0, 2, 0), (0, 0, 2), (0, 2, 2)
        fu, fd = (1, 2, 1), (1, 1, 2)
        return {
            ((e, e, e), (e, e, e)): 1.,
            ((f, f, f2), (f2, f, f)): 1.,
            ((u, u, u2), (u2, u, u)): 1.,
            ((d, d, d2), (d2, d, d)): 1.,

            ((u, e, u), (u, e, u)): ct,
            ((d, e, d), (d, e, d)): ct,
            ((e, u, u), (u, u, e)): ct,
            ((e, d, d), (d, d, e)): ct,
            ((u, e, u), (u, u, e)): st,
            ((d, e, d), (d, d, e)): st,
            ((e, u, u), (u, e, u)): -st,
            ((e, d, d), (d, e, d)): -st,

            ((f, u, fu), (fu, u, f)): ct,
            ((f, d, fd), (fd, d, f)): ct,
            ((u, f, fu), (fu, f, u)): ct,
            ((d, f, fd), (fd, f, d)): ct,
            ((u, f, fu), (fu, u, f)): st,
            ((d, f, fd), (fd, d, f)): st,
            ((f, u, fu), (fu, f, u)): -st,
            ((f, d, fd), (fd, f, d)): -st,

            ((f, e, f), (f, e, f)): ct * ct,
            ((f, e, f), (f, d, u)): ct * st,
            ((f, e, f), (f, u, d)): -ct * st,
            ((f, e, f), (f, f, e)): st * st,

            ((e, f, f), (f, f, e)): ct * ct,
            ((e, f, f), (f, u, d)): ct * st,
            ((e, f, f), (f, d, u)): -ct * st,
            ((e, f, f), (f, e, f)): st * st,

            ((u, d, f), (f, d, u)): ct * ct,
            ((u, d, f), (f, u, d)): st * st,
            ((u, d, f), (f, e, f)): -ct * st,
            ((u, d, f), (f, f, e)): ct * st,

            ((d, u, f), (f, u, d)): ct * ct,
            ((d, u, f), (f, d, u)): st * st,
            ((d, u, f), (f, e, f)): ct * st,
            ((d, u, f), (f, f, e)): -ct * st,
        }

    def valuesSU2():
        # Occupations of one spatial orbital in U(1) x SU(2)
        raise NotImplementedError
        # e, s, f = (0, 0, 0), (1, 1, 1), (0, 2, 0)

    if tuple(symmetries) == ('fermionic', 'U(1)', 'U(1)'):
        values = valuesU1()
    elif tuple(symmetries) == ('fermionic', 'U(1)', 'SU(2)'):
        values = valuesSU2()
    else:
        raise NotImplementedError(
            f'rotation Tensor not implemented for {symmetries}')

    for k, v in values.items():
        U[k] = np.array(v, ndmin=4)

    return U, npl

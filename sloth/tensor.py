from sloth.symmetries import _SYMMETRIES
import jax.numpy as np


class InternalNode():
    """Internal nodes for big tensors. Needed for the SU(2) symmetry
    """
    def __init__(self, tens):
        if not isinstance(tens, Tensor):
            raise TypeError(f'{tens} should be off class {Tensor.__name__}')
        self.tensor = tens


class Leg:
    """
    It is mutable though

    Maybe make it indexable for the symmetry sectors.
    and iterable
    Maybe give with it also the dimension.
    """
    def __init__(self, begin, end):
        self._begin = begin
        self._end = end

    @property
    def begin(self):
        """I do not want people to foefel with this
        """
        return self._begin

    @property
    def end(self):
        """I do not want people themselves to foefel with this
        """
        return self._end

    @property
    def internal(self):
        """Boolean if leg is internal or not
        """
        return isinstance(self.begin, InternalNode) or \
            isinstance(self.end, InternalNode)

    def __repr__(self):
        # This is custom since I do not want that the bond begin or end are
        # completely expanded if they are Tensors.
        def default_expr(obj):
            return f"{str(obj.__class__)[8:-2]} object at {hex(id(obj))}"

        def repr_node(x):
            return f"<{default_expr(x)}>" if isinstance(x, Tensor) else f"{x}"

        return f"<{default_expr(self)}:" + \
            f"({repr_node(self.begin)}, {repr_node(self.end)})>"

    def __iter__(self):
        # So you can use `in`
        yield self.begin
        yield self.end

    def ingoing(self, obj):
        if obj != self.begin and obj != self.end:
            raise ValueError(f'{obj} is not connected to bond')
        return self.end == obj

    def substitute(self, obj, nobj):
        """Substitutes the obj in the leg by nobj
        """
        if self.begin == obj:
            self._begin = nobj
        if self.end == obj:
            self._end = nobj


class Tensor:
    """
    Attrs:
        coupling: tuple((leg1, leg2, leg3), ...)
        maybe go to ndarray?
        for the different coupling

        sorted list of symmetries
    """
    def __init__(self, symmetries, coupling=None):
        invalids = [i for i in symmetries if i not in _SYMMETRIES]
        if invalids:
            raise ValueError(f'Invalid symmetries inputted: {invalids}')

        symmetries = list(symmetries)
        symmetries.sort(reverse=True)
        self._symmetries = tuple(symmetries)
        self._coupling = None if coupling is None else \
            tuple(tuple(c) for c in coupling)
        self._data = {}

    @property
    def coupling(self):
        return self._coupling

    @coupling.setter
    def coupling(self, coupling):
        if isinstance(coupling[0], (list, tuple)):
            if sum([len(c) == 3 for c in coupling]) != len(coupling):
                raise ValueError('coupling should be an (x, 3) (3) nested '
                                 f'list/tuple. Not {coupling}.')
            self._coupling = tuple(tuple(c) for c in coupling)
        else:
            if len(coupling) != 3:
                raise ValueError('coupling should be an (x, 3) or (3) nested'
                                 f'list/tuple. Not {coupling}.')
            self._coupling = (tuple(coupling),)

    @property
    def symmetries(self):
        return self._symmetries

    @property
    def flattened_coupling(self):
        return [el for c in self.coupling for el in c]

    @property
    def outerlegs(self):
        return [x for x in self.flattened_coupling if not x.internal]

    @property
    def internallegs(self):
        return [x for x in self.flattened_coupling if x.internal]

    @property
    def neighbours(self):
        return list(set(x for leg in self.outerlegs for x in leg
                        if x != self and isinstance(x, Tensor)))

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

    def coupling_id(self, bond):
        """Finds first occurence of bond in the coupling
        """
        for i, x in enumerate(self.coupling):
            for j, y in enumerate(x):
                if y == bond:
                    return i, j
        return None

    def connections(self, B):
        """Returns the connections between `Tensor` `self` and `B` and thus if
        they can be contracted along these connections.
        """
        if not isinstance(B, Tensor):
            raise TypeError(f'{B} is not of {self.__class__}')

        return [x for x in self.flattened_coupling
                if x in B.flattened_coupling]

    def substituteleg(self, oldleg, newleg):
        """returns a coupling tuple where oldleg in self is swapped with
        newleg.
        """
        if oldleg not in self.flattened_coupling:
            raise ValueError(f'{oldleg} not a coupling for {self}')
        return tuple(tuple(newleg if x == oldleg else x for x in y) for y in
                     self.coupling)

    def __matmul__(self, B):
        """Trying to completely contract self and B for all matching bonds.
        """
        connections = self.connections(B)
        if not connections:
            raise ValueError(f'No connections found between {self} and {B}')

        raise NotImplementedError

    def qr(self, bond):
        """Executes a QR decomposition for one of the bonds.

        This bond can not be an internal bond, no prefactors from symmetry
        sectors needed.

        bond is ingoing:
            R will couple as (old bond, vacuum -> new bond)
            new bond is R -> Q
        bond is outgoing:
            R will couple as (new bond, vacuum -> old bond)
            new bond is Q -> R

        self is appropriately changed
        """
        if bond.internal:
            raise ValueError(f'{bond} is internal.')

        ingoing = bond.ingoing(self)
        i1, i2 = self.coupling_id(bond)
        f_id = self.flattened_coupling.index(bond)

        R = Tensor(self.symmetries)
        bond.substitute(self, R)

        newbond = Leg(R, self) if ingoing else Leg(self, R)
        R.coupling = (bond, Leg(None, R), newbond) if ingoing else \
            (newbond, Leg(None, R), bond)
        self.coupling = self.substituteleg(bond, newbond)
        vacuum = (0,) * len(self.symmetries)

        assert [x == newbond for x in R.connections(self)] == [True]

        keys = set([k[i1][i2] for k in self])
        transp = list(range(len(self.outerlegs)))
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
            R[((key, vacuum, key),)] = r.T if ingoing else r

            # moving back all the blocks into the original tensor
            for block, x in zip(blocks, np.split(q, Aspl)):
                self[block] = np.transpose(
                    x.reshape(np.array(self[block].shape)[transp]), i_transp)
        return R

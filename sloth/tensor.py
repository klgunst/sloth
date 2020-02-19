from sloth.symmetries import _SYMMETRIES
import jax
import jax.numpy as np
import collections


class SymKey(np.ndarray):
    def __hash__(self):
        hash(self.tostring())


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


class NestedTuple(tuple):
    """Sentinel is iterable objects that should not be converted in nested
    tuples.
    """
    def __new__(self, a, _init=True):
        if _init:
            return NestedTuple(a, _init=False)
        elif isinstance(a, (list, tuple)):
            return tuple.__new__(NestedTuple,
                                 (NestedTuple(ax, _init=False) for ax in a))
        else:
            return a

    def flatten(self):
        """Flattens the NestedTuple to a list.
        """
        def yielder(x):
            for y in x:
                if isinstance(y, NestedTuple):
                    yield from yielder(y)
                else:
                    yield y
        return [y for y in yielder(self)]

    def __getitem__(self, index):
        if isinstance(index, int):
            return self[[index]]
        elif not isinstance(index, collections.Iterator):
            return self[iter(index)]
        else:
            i = next(index)
            if not isinstance(i, int):
                raise ValueError(f'Only integers for indexing. {i} invalid')
            inner = tuple.__getitem__(self, i)
            if isinstance(inner, NestedTuple):
                try:
                    return inner[index]
                except StopIteration:
                    # indexing ended
                    return inner
            else:
                try:
                    next(index)
                    raise ValueError('Bottom of nested tuple reached')
                except StopIteration:
                    return inner

    def find(self, obj):
        """Find the object in the nested tuple and returns its indexes for the
        first occurence.
        """
        for i, x in enumerate(self):
            if x == obj:
                return (i,)
            elif isinstance(x, NestedTuple):
                result = x.find(obj)
                if result is not None:
                    return (i, *result)
        return None

    def substitute(self, obj, new_obj):
        """Substitute all occurences of obj with new_obj.
        """
        return NestedTuple([
            NestedTuple(new_obj, _init=False) if x == obj else
            (x if not isinstance(x, NestedTuple) else
             x.substitute(obj, new_obj)) for x in self]
                           )


class Tensor:
    """
    Attrs:
        couplings: NestedTuple((leg1, leg2, leg3), ...)
        maybe go to ndarray?
        for the different couplings

        sorted list of symmetries
    """
    def __init__(self, symmetries, couplings=None):
        invalids = [i for i in symmetries if i not in _SYMMETRIES]
        if invalids:
            raise ValueError(f'Invalid symmetries inputted: {invalids}')

        symmetries = list(symmetries)
        symmetries.sort(reverse=True)
        self._symmetries = tuple(symmetries)
        self._couplings = None if couplings is None else \
            NestedTuple(couplings)
        self._data = {}

    @property
    def couplings(self):
        return self._couplings

    @couplings.setter
    def couplings(self, couplings):
        self._couplings = NestedTuple(couplings)

    @property
    def symmetries(self):
        return self._symmetries

    @property
    def outerlegs(self):
        return [x for x in self.couplings.flatten() if not x.internal]

    @property
    def internallegs(self):
        return [x for x in self.couplings.flatten() if x.internal]

    @property
    def neighbours(self):
        return [x.begin if x.begin != self else x.end for x in self.outerlegs]

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

    def connections(self, B):
        """Returns the connections between `Tensor` `self` and `B` and thus if
        they can be contracted along these connections.
        """
        if not isinstance(B, Tensor):
            raise TypeError(f'{B} is not of {self.__class__}')

        return [x for x in self.couplings.flatten()
                if x in B.couplings.flatten()]

    def swapped_leg(self, oldleg, newleg):
        """returns a couplings tuple where oldleg in self is swapped with
        newleg.
        """
        if oldleg not in self.couplings.flatten():
            raise ValueError(f'{oldleg} not a coupling for {self}')

        return tuple(tuple(y if y != oldleg else newleg for y in x)
                     for x in self.couplings)

    def __matmul__(self, B):
        """Trying to completely contract self and B for all matching bonds.
        """
        connections = self.connections(B)
        if not connections:
            raise ValueError(f'No connections found between {self} and {B}')

        raise NotImplementedError

    def qr(self, bond):
        """Executes a QR decomposition for one of the bonds.

        This bond can not be an internal bond.

        bond is ingoing:
            R will couple as (old bond, vacuum -> new bond)
            new bond is R -> Q
        bond is outgoing:
            R will couple as (new bond, vacuum -> old bond)
            new bond is Q -> R
        """
        if bond.internal:
            raise ValueError(f'{bond} is internal.')

        ingoing = bond.end == self

        R = Tensor(self.symmetries)
        Q = Tensor(self.symmetries)
        newbond = Leg(R, Q) if ingoing else Leg(Q, R)
        R.coupling = ((bond, Leg('Vacuum', R), newbond),) if ingoing else \
            ((newbond, Leg('Vacuum', R), bond),)
        Q.coupling = self.swapped_leg(bond, newbond)

        assert [x == newbond for x in R.connections(Q)] == [True]

        ind = self.index(bond)
        flat_ind = self.flattened_index(bond)
        keys = set([k[ind[0]][ind[1]] for k in self])
        transp = list(range(self.dims))
        transp.pop(flat_ind)
        transp.append(flat_ind)
        transp = np.array(transp)

        inflow = self.flow[ind[0]][ind[1]]

        for key in keys:
            blocks = [k for k in self if k[ind[0]][ind[1]] == key]

            leading_dim = set(self[k].shape[flat_ind] for k in blocks)
            # check if dimension is consistent everywhere
            assert len(leading_dim) == 1
            leading_dim = leading_dim.pop()

            size = sum(self[k].size for k in blocks)
            assert size % leading_dim == 0
            other_dim = size // leading_dim

            A = np.zeros((other_dim, leading_dim))
            # Moving all needed blocks into one matrix
            indx = 0
            for block in blocks:
                AT = np.transpose(self[block], transp).reshape(-1, leading_dim)
                slice_of_cake = jax.ops.index[indx:AT.shape[0] + indx, :]
                A = jax.ops.index_update(A, slice_of_cake, AT)
                indx += AT.shape[0]

            q, r = np.linalg.qr(A)
            R[((key, key),)] = r.T if inflow else r

            # moving back all the blocks into the original tensor
            indx = 0
            for block in blocks:
                oshape = np.array(self[block].shape)[transp]
                self[block] = np.transpose(
                    q[indx:np.prod(oshape[:-1]) + indx, :].reshape(oshape),
                    np.argsort(transp)
                )
                indx += AT.shape[0]

        self.swap_index(bond, R)
        if isinstance(bond, Tensor):
            # Swap the bond index from self to R
            bond.swap_index(self, R)
        return R

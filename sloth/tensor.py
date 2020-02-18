from sloth.symmetries import _SYMMETRIES
import jax
import jax.numpy as np


class Tensor:
    def __init__(self, symmetries, flow=None, indexes=None):
        invalids = [i for i in symmetries if i not in _SYMMETRIES]
        if invalids:
            raise ValueError(f'Invalid symmetries inputted: {invalids}')

        self.symmetries = symmetries
        self.flow = flow
        self.indexes = indexes
        self._data = {}

    # prevent expanding the indices every time
    _in_repr = False

    def __repr__(self):
        s = f"<{str(self.__class__)[8:-2]} object at {hex(id(self))}>"
        if not Tensor._in_repr:
            Tensor._in_repr = True
            metadata = {k: v for k, v in self.__dict__.items() if k != '_data'}
            s += f"({metadata})\n"
            s += ''.join([f"{k}:\n{v}\n" for k, v in self.items()])
            s += "\n"
            Tensor._in_repr = False
        return s

    def __setitem__(self, index, value):
        self._data[index] = value

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self): return len(self._data)

    def __iter__(self):
        return self._data.__iter__()

    def __matmul__(self, B):
        if not isinstance(B, Tensor):
            raise ValueError(f'{B} is not of {self.__class__}')

        if self.dims == 2:
            return Tensor.TensMatmult(B, self)
        elif B.dims == 2:
            return Tensor.TensMatmult(self, B)
        else:
            raise NotImplementedError

    def TensMatmult(A, B):
        """For this case no symmetry blablabla needed

        Mat should be | >< |
        otherwise first swap around?
        destructive again
        """
        if B.dims != 2:
            raise ValueError
        if not B.flow[0][0] or B.flow[0][1]:
            raise ValueError('Should be | >< |')

        A_id, A_fid = A.index(B), A.flattened_index(B)
        B_fid = B.flattened_index(A)
        other_one = B.indexes[0][0 if B_fid == 1 else 1]
        for k in A:
            key = k[A_id[0]][A_id[1]]
            A[k] = np.tensordot(A[k], B[((key, key),)], axes=[A_fid, B_fid])
        A.swap_index(B, other_one)
        if isinstance(other_one, Tensor):
            other_one.swap_index(B, A)

        return A

    def items(self):
        return self._data.items()

    def swap_index(self, orig, new):
        self.indexes = tuple(
            tuple(new if x == orig else x for x in y) for y in self.indexes
        )

    def flattened_index(self, bond):
        """Needs to be a nested tuple
        """
        return [z for y in (x for x in self.indexes) for z in y].index(bond)

    def index(self, bond):
        for i, x in enumerate(self.indexes):
            for j, y in enumerate(x):
                if y == bond:
                    return (i, j)
        raise ValueError(f'{bond} not in list')

    @property
    def size(self):
        return sum([d.size for _, d in self._data.items()])

    @property
    def dims(self):
        return len([z for y in (x for x in self.indexes) for z in y])

    def qr(self, bond):
        """Executes a QR decomposition for one of the indices.
        """
        ind = self.index(bond)
        flat_ind = self.flattened_index(bond)
        keys = set([k[ind[0]][ind[1]] for k in self])
        transp = list(range(self.dims))
        transp.pop(flat_ind)
        transp.append(flat_ind)
        transp = np.array(transp)

        inflow = self.flow[ind[0]][ind[1]]
        R = Tensor(
            self.symmetries,
            ((True, False),),
            indexes=((bond if inflow else self, self if inflow else bond),)
        )

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



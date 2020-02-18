import jax.numpy as np
from jax import jit
import h5py
from sloth.symmetries import _SYMMETRIES
from sloth.tensor import Tensor


def read_h5(filename):
    """This reads a hdf5 file.

    Returns the root of the network.
    """
    h5file = h5py.File(filename, 'r')
    sym = tuple(_SYMMETRIES[i] for i in h5file['bookkeeper'].attrs['sgs'])

    sites = h5file['network'].attrs['sites'].item()
    tensors = [Tensor(sym, flow=((True, True, False),)) for i in range(sites)]
    bonds = np.array(h5file['network']['bonds']).reshape(-1, 2)
    sitetoorb = h5file['network']['sitetoorb']

    for tid, A in enumerate(tensors):
        T = h5file['T3NS'][f'tensor_{tid}']
        assert T.attrs['nrsites'] == 1
        assert T.attrs['sites'][0] == tid
        assert T.attrs['nrblocks'] == len(T['qnumbers'])

        neighbors = [None, None, None]
        symsecs = [None, None, None]

        for bid, bond in enumerate(bonds):
            if bond[0] == tid:
                assert neighbors[2] is None
                symsecs[2] = h5file['bookkeeper'][f'v_symsec_{bid}']
                if bond[1] == -1:
                    neighbors[2] = 'target'
                else:
                    neighbors[2] = tensors[bond[1]]
            if bond[1] == tid:
                index = 0 if neighbors[0] is None else 1
                assert neighbors[index] is None
                symsecs[index] = h5file['bookkeeper'][f'v_symsec_{bid}']
                if bond[0] == -1:
                    neighbors[index] = 'vacuum'
                else:
                    neighbors[index] = tensors[bond[0]]

        if sitetoorb[tid] != -1:
            assert neighbors[1] is None
            symsecs[1] = h5file['bookkeeper'][f'p_symsec_{sitetoorb[tid]}']
            neighbors[1] = str(sitetoorb[tid])

        A.indexes = (tuple(neighbors),)
        # reshaping the irreps bit
        sirr = [np.array(s['irreps']).reshape(
            s.attrs['nrSecs'].item(), -1)[:, :len(sym)] for s in symsecs
                ]

        @jit
        def get_ids(qn):
            dims = np.array([s.attrs['nrSecs'].item() for s in symsecs],
                            dtype=np.int32)
            divs = np.array([np.prod(dims[:i]) for i in range(len(dims))],
                            dtype=np.int32)

            indices = (qn // divs) % dims
            return indices

        block = T['block_0']
        for block_id in range(block.attrs['nrBlocks'].item()):
            indexes = get_ids(T['qnumbers'][block_id])
            shape = [s['dims'][i] for i, s in zip(indexes, symsecs)]
            key = (tuple([tuple(irr[i]) for i, irr in zip(indexes, sirr)]),)

            begin = block['beginblock'][block_id]
            end = block['beginblock'][block_id + 1]
            A[key] = np.array(block['tel'][begin:end]).reshape(shape)

    # root of the network
    return tensors[-1]

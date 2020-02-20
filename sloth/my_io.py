import jax.numpy as np
from jax import jit
import h5py
from sloth.symmetries import _SYMMETRIES
from sloth.tensor import Tensor, Leg


def read_h5(filename):
    """This reads a hdf5 file.

    Returns the root of the network.
    """
    h5file = h5py.File(filename, 'r')
    sym = [_SYMMETRIES[i] for i in h5file['bookkeeper'].attrs['sgs']]

    sites = h5file['network'].attrs['sites'].item()
    tensors = [Tensor(sym) for i in range(sites)]
    bonds = np.array(h5file['network']['bonds']).reshape(-1, 2)

    # make the virtual legs
    vlegs = [Leg(*[tensors[i] if i != -1 else None for i in x]) for x in bonds]

    # make the physical legs for each tensor (or none if virtual tensor)
    sitetoorb = h5file['network']['sitetoorb']
    bookie = h5file['bookkeeper']
    plegs = [Leg(f'p{i}', t) for i, t in zip(sitetoorb, tensors) if i != -1]

    for tid, A in enumerate(tensors):
        T = h5file['T3NS'][f'tensor_{tid}']
        assert T.attrs['nrsites'] == 1
        assert T.attrs['sites'][0] == tid
        assert T.attrs['nrblocks'] == len(T['qnumbers'])

        tbonds = [i for i, x in enumerate(bonds[:, 1]) if x == tid] + \
            ([] if sitetoorb[tid] == -1 else [f'p{sitetoorb[tid]}']) + \
            [i for i, x in enumerate(bonds[:, 0]) if x == tid]
        assert len(tbonds) == 3

        symsecs = tuple(bookie[f'v_symsec_{i}'] if isinstance(i, int) else
                        bookie[f'p_symsec_{i[1:]}'] for i in tbonds)

        A.coupling = tuple(vlegs[i] if isinstance(i, int) else
                           plegs[int(i[1:])] for i in tbonds)

        # No none's in coupling
        assert not [i for i in A.coupling[0] if i is None]

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

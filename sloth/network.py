from sloth.tensor import Leg, Tensor
import jax.numpy as np
from jax import jit
import h5py
from sloth.symmetries import _SYMMETRIES


class TNS:
    """Class for the TNS.

    Attrs:
        A dictionary which gives for each leg it's in and out.
    """
    def __init__(self):
        self._legs = {}

    def __setitem__(self, idx, value):
        self._legs[idx] = value

    def __getitem__(self, idx):
        return self._legs[idx]


def read_h5(filename):
    """This reads a hdf5 file.

    Returns the network.
    """
    tns = TNS()

    h5file = h5py.File(filename, 'r')
    sym = [_SYMMETRIES[i] for i in h5file['bookkeeper'].attrs['sgs']]

    sites = h5file['network'].attrs['sites'].item()
    tensors = [Tensor(sym) for i in range(sites)]
    bonds = np.array(h5file['network']['bonds']).reshape(-1, 2)

    # make the virtual legs
    vlegs = [Leg(x != -1, y != -1 and x != -1, x == -1) for x, y in bonds]
    plegs = [Leg() for i in range(h5file['network'].attrs['psites'].item())]
    for vl in vlegs:
        tns[vl] = [None, None]
    for i, pl in enumerate(plegs):
        tns[pl] = [f'p{i}', None]

    sitetoorb = h5file['network']['sitetoorb']
    bookie = h5file['bookkeeper']

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
        coupl = [vlegs[i] if isinstance(i, int) else
                 plegs[int(i[1:])] for i in tbonds]
        A.coupling = [(i, b) for i, b in zip(coupl, [True, True, False])]
        for x, y in A.coupling[0]:
            assert tns[x][y] is None
            tns[x][y] = A

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
    return tns

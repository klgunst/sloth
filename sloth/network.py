from sloth.tensor import Leg, Tensor
import jax.numpy as np
from jax import jit
import h5py
import networkx as nx
from sloth.symmetries import _SYMMETRIES


class TNS(nx.MultiDiGraph):
    """Class for the TNS.

    DiGraph subclassed

    Attrs:
        _legs: A dictionary which gives for each leg it's in and out.
    """
    def __init__(self):
        self._legs = {}

    def __setitem__(self, idx, value):
        self._legs[idx] = value

    def __getitem__(self, idx):
        return self._legs[idx]

    def __iter__(self):
        return self._legs.__iter__()

    def __len__(self):
        return len(self._legs)

    def gettensors(self):
        """Returns a set of all the tensors found in the network
        """
        return set(T for k, ll in self.items() for T in ll
                   if isinstance(T, Tensor))

    def getvirtuals(self):
        """Retrieves all the virtual Legs in the network
        """
        return set(k for k, (L, R) in self.items()
                   if isinstance(L, Tensor) and isinstance(R, Tensor))

    def physical_id(self, T):
        """Returns index of physical ids if tensor has physical connections.
        """
        return [int(self[i][not T.flowof(i)][1:]) for i in T.indexes if
                isinstance(self[i][not T.flowof(i)], str)]

    def items(self):
        return self._legs.items()

    def lasttensor(self):
        lasttensors = [v[0] for k, v in self.items() if v[1] is None]

        if len(lasttensors) != 1:
            raise ValueError('Multiple last tensors in network.')
        return lasttensors[0]

    def connections(self, tensor):
        """Finds all objects that this tensor is connected to.

        This can be a Tensor instance,
        or a string for physical indices
        or None for vacuums
        """
        return [self[i][not tensor.flowof(i)] for i in tensor.indexes]

    def tneighbours(self, tensor):
        """Finds all other tensors this tensor is connected to in the network.
        """
        return [T for T in self.connections(tensor) if isinstance(T, Tensor)]

    def adjoint(self):
        """Returns a deep copy with the adjoint of the tns.
        """
        raise NotImplementedError

    def complete_contract(self):
        T = self.lasttensor()
        ready = set(self.tneighbours(T))
        added = set(T)

        while True:
            try:
                A = ready.pop()
            except KeyError:
                break

            assert A not in added
            T = A @ T
            added.add(A)
            ready.update([t for t in self.tneighbours(A) if t not in added])
        return T

    def plotGraph(self, ax=None, with_id=False):
        """Plots a graph to the current pyplot scope.
        """
        import networkx as nx

        if not hasattr(self, '_G'):
            self._G = nx.DiGraph()
            sites = self.gettensors()
            virtuals = self.getvirtuals()

            self._G.add_nodes_from(sites)
            self._G.add_edges_from([self[i] for i in virtuals])

        edge_labels = None

        color_map = ['lightblue' if len(self.physical_id(s)) == 1 else 'green'
                     for s in self._G]

        pos = None
        if 'green' not in color_map:
            count = 0
            pos = {}
            prev = None
            for edge in self._G.edges:
                assert prev is None or prev == edge[0]
                pos[edge[0]] = (count, 0)
                count += 1
                prev = edge[1]
            pos[prev] = (count, 0)

        labels = {}
        for s in sites:
            pid = self.physical_id(s)
            assert len(pid) < 2
            if len(pid) == 1:
                labels[s] = pid[0]

        nx.draw(self._G, pos=pos, node_color=color_map, ax=ax,
                labels=labels, edge_labels=edge_labels, with_labels=True)


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
    vlegs = [Leg(phase=x != -1, pref=(y != -1 and x != -1), vacuum=x == -1)
             for x, y in bonds]
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

    return tns

from sloth.tensor import Leg, Tensor
import networkx as nx
import jax.numpy as np
from jax import jit
import h5py
from sloth.symmetries import _SYMMETRIES


class TNS(nx.MultiDiGraph):
    """Class for the TNS.

    MultiGraph subclassed

    Attrs:
        _loose_legs: dictionary with Leg instances as keys. The values are the
        nodes (Tensors) to which they are connected. These encompasses the
        loose legs of the network, these are not yet added to the
        Multigraph's edges.
    """
    def __init__(self, data=None, **attr):
        super(TNS, self).__init__(**attr)

        self._loose_legs = {}
        if data is not None:
            self.add_nodes_from(data, **attr)

        if isinstance(data, TNS):
            self._loose_legs = data._loose_legs.copy()

    def add_nodes_from(self, nodes, **attr):
        for n in nodes:
            if isinstance(n, tuple) and len(n) == 2 and isinstance(n[1], dict):
                self.add_node(n[0], **n[1], **attr)
            else:
                self.add_node(n, **attr)

    def add_node(self, node, **attr):
        if not isinstance(node, Tensor):
            raise ValueError('node should be a Tensor instance.')

        super(TNS, self).add_node(node, **attr)
        # possibly connects the added tensor to a tensor already in the network
        # if they have same Legs.
        for ll in node.indexes:
            if ll in self._loose_legs:
                inflow = node.flowof(ll)
                other_node, _ = self._loose_legs.pop(ll)
                if inflow == other_node.flowof(ll):
                    raise ValueError('Can\'t add node, conflicting flow')

                super(TNS, self).add_edge(other_node if inflow else node,
                                          node if inflow else other_node,
                                          leg=ll, label=hex(id(ll)))
            else:
                # Node and no name
                self._loose_legs[ll] = node, None

    def remove_nodes_from(self, nodes):
        for n in nodes:
            self.remove_node(n)

    def remove_node(self, node):
        super(TNS, self).remove_node(node)
        self._loose_legs = {k: (n, name) for k, (n, name) in
                            self._loose_legs.items() if n != node}

    def name_loose_edges_from(self, ebunch, **attr):
        for e in ebunch:
            u, v = e[0:2]
            self.name_loose_edge(u, v, **attr)

    def name_loose_edge(self, leg, name, **attr):
        """This is only for edges which do not connect Tensor instances.

        Args:
            leg: A Leg instance that should already be added in the
            _loose_legs attribute of self.
            name: The name of the loose edge.
        """
        self._loose_legs[leg] = (self._loose_legs[leg][0], name)

    def fancy_draw(self, ax=None, node_color=None):
        """Does whole bunch of stuff.

        Cast the current MultiGraph to a MultiDiGraph
        """
        from networkx.algorithms.shortest_paths.unweighted \
            import single_source_shortest_path_length as ss_short_path_length
        from networkx.algorithms.shortest_paths import shortest_path
        from networkx.algorithms.distance_measures import extrema_bounding

        pos, color_map, labels = None, None, None

        # If it is the TNS itself I want to draw
        undirected = self.to_undirected()
        if nx.algorithms.tree.recognition.is_tree(undirected):
            labels = {}
            for ll, (N, name) in self._loose_legs.items():
                if name:
                    if N in labels:
                        labels[N] += f',{name[1:]}'
                    else:
                        labels[N] = name[1:]

            color_map = ['lightblue' if s in labels else 'green' for s in self]

            # make a new graph keeping all branching tensors and edge tensors
            for s in labels:
                neighbors = list(undirected.neighbors(s))
                assert len(neighbors) <= 2
                if len(neighbors) == 2:
                    undirected.add_edge(*neighbors)
                    undirected.remove_node(s)

            # choose a branching tensor as center or else the last center
            radius = extrema_bounding(undirected, compute='radius')
            center = extrema_bounding(undirected, compute='center')[0]

            # Assign every node to the right shell
            nlist = [[] for i in range(radius + 1)]
            for k, v in ss_short_path_length(undirected, center).items():
                nlist[v].append(k)
            # First shell exists out of one Node
            assert len(nlist[0]) == 1

            # create array of powers of radiuses
            pos = {nlist[0][0]: {'r': 0, 'angle': 0}}
            rs = np.power(np.sqrt(2), np.arange(radius))
            nodes_on_shell = 3
            for shell, r in zip(nlist[:-1], rs):
                step = 2 * np.pi / nodes_on_shell
                nodes_on_shell *= 2
                for node in shell:
                    neighbors = list(undirected.neighbors(node))
                    angle = pos[node]['angle'] if len(neighbors) != 3 else \
                        pos[node]['angle'] - step / 2.
                    for neighbor in neighbors:
                        if neighbor not in pos:
                            pos[neighbor] = {'r': r, 'angle': angle}
                            angle += step

            def pol2cart(r=0, angle=0):
                return np.array([r * np.cos(angle), r * np.sin(angle)])

            pos = {k: pol2cart(**v) for k, v in pos.items()}
            for u, v, _ in undirected.edges:
                nodes_inbetween = shortest_path(
                    self.to_undirected(as_view=True), u, v)
                xy_step = (pos[v] - pos[u]) / (len(nodes_inbetween) - 1.)

                for ii, nodes in enumerate(nodes_inbetween[1:-1], start=1):
                    pos[nodes] = pos[u] + ii * xy_step

        if node_color is None:
            node_color = color_map

        nx.draw(self, pos=pos, ax=ax, node_color=node_color, labels=labels,
                node_size=[len(s.coupling) * 300 for s in self],
                with_labels=True)

    @property
    def sink(self):
        """Returns the sink tensor.
        """
        return nx.algorithms.dag.dag_longest_path(self)[-1]

    def contracted_nodes(self, A, B):
        """Contracts two nodes of a given graph and returns the resulting
        graph.
        """
        if A not in nx.all_neighbors(self, B):
            raise ValueError(
                'The two tensors are not neighbours in the network')

        T = A @ B
        result = TNS(tuple(T if n == A else n for n in self if n != B))

        result.name_loose_edges_from([[ll, name] for ll, (_, name) in
                                      self._loose_legs.items()])
        return result

    def boundaries(self):
        """iterator over all the boundary tensors (only if it is a dag)
        """
        undirected = self.to_undirected(as_view=True)
        if not nx.algorithms.tree.recognition.is_tree(undirected):
            raise ValueError('Graph needs to be a a tree')

        for n in undirected:
            if undirected.degree(n) == 1:
                yield n

    def contract(self, pass_intermediates=False):
        """Fully contracts the network. Is not necessarily efficient.
        """
        netw = TNS(self)
        if pass_intermediates:
            interm = []

        while netw.number_of_nodes() != 1:
            T = None
            for x in netw.boundaries():
                if T is None or len(T.indexes) > len(x.indexes):
                    T = x

            # One of the neighbours
            for A in nx.all_neighbors(netw, T):
                break
            netw = netw.contracted_nodes(A, T)
            if pass_intermediates:
                interm.append(netw)

        if pass_intermediates:
            return netw, interm
        else:
            return netw

    def qr(self, node, edge):
        """Splits a node into two subnodes.

        Edge decides how to split the node.
        """
        tns = TNS(T for T in self if T != node)
        tns.add_nodes_from(node.qr(edge['leg']))
        tns.name_loose_edges_from([[ll, name] for ll, (_, name) in
                                   self._loose_legs.items()])
        return tns


def read_h5(filename):
    """This reads a hdf5 file.

    Returns the network.
    """
    h5file = h5py.File(filename, 'r')
    sym = [_SYMMETRIES[i] for i in h5file['bookkeeper'].attrs['sgs']]

    sites = h5file['network'].attrs['sites'].item()
    tensors = [Tensor(sym) for i in range(sites)]
    bonds = np.array(h5file['network']['bonds']).reshape(-1, 2)

    # make the virtual legs
    vlegs = [Leg(phase=x != -1, pref=(y != -1 and x != -1), vacuum=x == -1)
             for x, y in bonds]
    plegs = [Leg() for i in range(h5file['network'].attrs['psites'].item())]

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

    tns = TNS(tensors)
    tns.name_loose_edges_from([[pl, f'p{ii}'] for ii, pl in enumerate(plegs)])

    assert nx.algorithms.dag.is_directed_acyclic_graph(tns)
    return tns

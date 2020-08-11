from sloth.tensor import Leg, Tensor, _SYMMETRIES
import networkx as nx
import numpy as np
import h5py


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

    @property
    def unique_legs(self):
        return set(x[-1] for x in self.edges(data='leg'))

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

            color_map = ['lightblue' if self.degree(s) < 3 else 'green'
                         for s in self]

            # make a new graph keeping all branching tensors and edge tensors
            for s in self:
                neighbors = list(undirected.neighbors(s))
                if undirected.degree(s) == 2:
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
    def orbitals(self):
        """The orbitals in the current tensor network.
        """
        return set(int(name[1:]) for k, (_, name) in self._loose_legs.items()
                   if name is not None and name[0] == 'p')

    @property
    def sink(self):
        """Returns the sink tensor.
        """
        return nx.algorithms.dag.dag_longest_path(self)[-1]

    def nodes_with_leg(self, leg):
        """Returns the two nodes that border on the given leg.
        """
        for u, v, ll in self.to_undirected(as_view=True).edges(data='leg'):
            if ll == leg:
                return u, v

    def contracted_nodes(self, A, B):
        """Contracts two nodes of a given graph and returns the resulting
        graph.

        If singular values live on the leg in between than these are also
        contracted
        """
        if A not in nx.all_neighbors(self, B):
            raise ValueError(
                'The two tensors are not neighbours in the network')

        data = self.to_undirected(as_view=True).edges[A, B, 0]
        if 'singular values' in data:
            C = A @ data['singular values']
        else:
            C = A
        T = C @ B
        result = TNS(tuple(T if n == A else n for n in self if n != B))

        result.name_loose_edges_from([[ll, name] for ll, (_, name) in
                                      self._loose_legs.items() if name])

        for u, v, k, data in self.edges(data=True, keys=True):
            if 'singular values' in data:
                if u in (A, B) and v in (A, B) and k == 0:
                    # This singular values should not be added
                    continue

                u = T if u in (A, B) else u
                v = T if v in (A, B) else v
                result.edges[u, v, k]['singular values'] = \
                    data['singular values']

        return result, T

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
        netw = self
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
            netw = netw.contracted_nodes(A, T)[0]
            if pass_intermediates:
                interm.append(netw)

        if pass_intermediates:
            return netw, interm
        else:
            return netw

    def qr(self, node, leg, intermittent_renorm=False):
        """Splits a node into two subnodes.

        Edge decides how to split the node.
        """
        tns = TNS(T for T in self if T != node)
        Q, R = node.qr(leg)
        tns.add_nodes_from((Q, R))
        tns.name_loose_edges_from([[ll, name] for ll, (_, name) in
                                   self._loose_legs.items()])
        if intermittent_renorm:
            R /= np.linalg.norm(R.ravel())
        return tns, (Q, R)

    def svd(self, node, leg):
        """Does an SVD on a node along the internal leg.

        The singular values are saved on the leg itself and also returned.
        """
        tns = TNS(T for T in self if T != node)
        U, S, V = node.svd(leg, compute_uv=True)
        tns.add_nodes_from((U, V))

        # TODO: Should do this in another way
        tns.edges[U, V, 0]['singular values'] = S

        tns.name_loose_edges_from([[ll, name] for ll, (_, name) in
                                   self._loose_legs.items()])
        return tns, U, S, V

    def move_orthogonality_center(self, nodeA, nodeB, pass_interm=False,
                                  hook=None):
        """Moves the orthogonality center from nodeA to nodeB.

        Does not check if nodeA is orthogonality center!

        Args:
            intermittent_renorm: True if at every QR decomposition R should be
            renormalized.
        """
        interm = []
        path = nx.algorithms.shortest_paths.shortest_path(
            self.to_undirected(as_view=True), nodeA, nodeB)
        tns = self
        A = nodeA

        for B in path[1:]:
            leg = tns.to_undirected(as_view=True).edges[A, B, 0]['leg']
            tns2, (Q, R) = tns.qr(A, leg)
            tns, A = tns2.contracted_nodes(R, B)
            if hook:
                hook(tns, A, Q)

            if pass_interm:
                interm.extend([tns2, tns])

        if pass_interm:
            return tns, A, interm
        else:
            return tns, A

    def calculate_singular_values(self, current_ortho=None):
        """Calculates the singular values along **all** internal edges.

        If current orthogonality center is not given it is assumed to be the
        sink.
        """
        if not current_ortho:
            current_ortho = self.sink

        svals = {}
        legs_to_visit = self.unique_legs

        def go_through(tns, A):
            for x, y, leg in \
                    tns.to_undirected(as_view=True).edges(A, data='leg'):
                assert A == x
                if leg in legs_to_visit:
                    legs_to_visit.remove(leg)
                    cur_tns, (Q, R) = tns.qr(A, leg)
                    svals[leg] = R.svd(compute_uv=False)
                    cur_tns, B = cur_tns.contracted_nodes(R, y)
                    go_through(cur_tns, B)

            for leg in [k for k, v in tns._loose_legs.items() if v[0] == A]:
                _, (_, R) = tns.qr(A, leg)
                svals[leg] = R.svd(compute_uv=False)

        go_through(self, current_ortho)
        return svals

    def get_orbital_partition(self, leg):
        """Returns two sets one set for the collection of orbitals on each side
        of the leg in the network.
        """
        from networkx.algorithms.components import is_connected, \
            number_connected_components, connected_components

        tns = self.to_undirected(as_view=False)
        if not is_connected(tns):
            raise ValueError(
                'The given graph should be connected for a good partition')

        if leg in self._loose_legs:
            name = self._loose_legs[leg][1]
            if name and name[0] == 'p':
                orb = int(name[1:])

                lset = set([orb])
                rset = self.orbitals
                rset.remove(orb)
            else:
                lset = set()
                rset = self.orbitals
        else:
            tns.remove_edge(*self.nodes_with_leg(leg))
            assert number_connected_components(tns) == 2

            lset, rset = [
                set(int(name[1:]) for _, (n, name) in self._loose_legs.items()
                    if n in G and name and name[0] == 'p')
                for G in connected_components(tns)]

        assert lset.isdisjoint(rset)
        assert lset.union(rset) == self.orbitals
        return lset, rset

    def disentangle(self, node_iterator=None):
        """General swapping of indexes of two neighbouring tensors.

        node_iterator should return the neighbouring nodes to contract every
        step. If none it just sweeps and does whatever it feels like
        """
        raise NotImplementedError
        tns = self
        # if node_iterator is None:
        #    node_iterator = _simple_sweep_model

        for A, B in node_iterator(tns):
            tns, T = tns.contracted_nodes(A, B)

    def two_rdms(self, legs, current_ortho=None):
        """Calculates the two-orbital rdms of the network.

        Silly implementation
        """
        rdms = None
        legmap = {l: Leg(leg=l) for l in legs}

        def hook(tns, A, Q):
            nonlocal rdms
            nonlocal legmap

            common_leg = A.connections(Q).pop()
            Qa = Q.adj(common_leg)
            # Mapping for the legs
            Qaset = set(Qa.indexes).difference(Q.indexes)
            assert len(Qaset) == 1
            legmap[common_leg] = Qaset.pop()
            Qa.swaplegs(legmap)

            if rdms is None:
                # First start of calc
                assert legs[0] in Q.indexes
                # Change physical leg of adjoint
                rdms = (Q @ Qa).swap(legmap[common_leg], legs[0])
            else:
                assert rdms.coupling_id(legs[0])[0] \
                    == rdms.coupling_id(legmap[legs[0]])[0]
                rdms @= Q
                rdms @= Qa
            assert len(rdms.coupling) == 2

            if legs[1] in A.indexes:
                rdms @= A
                Aa = A.adj(leg=None).swaplegs(legmap)
                rdms @= Aa
                rdms = rdms.swap(legs[0], legmap[legs[1]])

        A = self._loose_legs[legs[0]][0]
        current_ortho = self.sink if current_ortho is None else current_ortho
        tns, X = self.move_orthogonality_center(current_ortho, A)
        B = tns._loose_legs[legs[1]][0]
        tns.move_orthogonality_center(X, B, hook=hook)
        return rdms


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
    vlegs = [Leg(vacuum=x == -1) for x, _ in bonds]
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
        isSink = bonds[tbonds[-1], 1] == -1
        assert len(tbonds) == 3

        symsecs = tuple(bookie[f'v_symsec_{i}'] if isinstance(i, int) else
                        bookie[f'p_symsec_{i[1:]}'] for i in tbonds)
        coupl = [vlegs[i] if isinstance(i, int) else
                 plegs[int(i[1:])] for i in tbonds]
        A.coupling = [(i, b) for i, b in zip(coupl, [True, True, False])]

        if isSink:
            SU2_ids = []
        else:
            SU2_ids = A.getSymmIds('SU(2)')

        def prefactor(key):
            return np.prod([np.sqrt(key[0][2][i] + 1) for i in SU2_ids])

        # reshaping the irreps bit
        sirr = [np.array(s['irreps']).reshape(
            s.attrs['nrSecs'].item(), -1)[:, :len(sym)] for s in symsecs]

        def get_ids(qn):
            dims = np.array([s.attrs['nrSecs'].item() for s in symsecs],
                            dtype=np.int32)
            divs = np.array([np.prod(dims[:i]) for i in range(len(dims))],
                            dtype=np.int32)

            indices = (qn // divs) % dims
            return indices

        block = T['block_0']
        for bid, (begin, end) in \
                enumerate(zip(block['beginblock'], block['beginblock'][1:])):
            indexes = get_ids(T['qnumbers'][bid])
            shape = [s['dims'][i] for i, s in zip(indexes, symsecs)]
            key = (tuple([tuple(irr[i]) for i, irr in zip(indexes, sirr)]),)

            # Have to watch out for that sneaky Fortran order!!
            Ablock = block['tel'][begin:end].reshape(shape, order='F')
            # And now cast everything to a C order and to np.float64
            A[key] = prefactor(key) * \
                np.array(Ablock, order='C', dtype=np.float64)

    tns = TNS(tensors)
    tns.name_loose_edges_from([[pl, f'p{ii}'] for ii, pl in enumerate(plegs)])

    assert nx.algorithms.dag.is_directed_acyclic_graph(tns)
    return tns

from sloth.tensor import Leg, Tensor, _SYMMETRIES
import networkx as nx
import numpy as np


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

        self._orthoCenter = None
        self._loose_legs = {}
        if data is not None:
            self.add_nodes_from(data, **attr)

        if isinstance(data, TNS):
            self._loose_legs = data._loose_legs.copy()
            self._orthoCenter = data._orthoCenter

    @property
    def orthoCenter(self):
        return self._orthoCenter

    @property
    def isMPS(self):
        """Checks if it is an MPS (otherwise a TTNS)
        """
        return len([b for b in self.boundaries()]) == 2

    def topologicalPsiteSort(self):
        """Only unique for MPS otherwise, branches can be swapped
        """
        pLegs = self.getPhysicalLegs()
        X = [pLegs.intersection(N.indexes) for N in nx.topological_sort(self)]
        return [x.pop() for x in X if x]

    def lexicographicalPsiteSort(self):
        """Returns the legs in p1, p2, p3, ... order
        """
        pLegs = self.getPhysicalLegs()
        return sorted(pLegs, key=lambda x: int(self._loose_legs[x][1][1:]))

    @property
    def unique_legs(self):
        return set(x[-1] for x in self.edges(data='leg'))

    @property
    def max_bondDimsension(self):
        return max(max(A.shape) for A in self)

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
            if self.orthoCenter in self:
                color_map[list(self).index(self.orthoCenter)] = 'grey'

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

    def getPhysicalLegs(self):
        return set(k for k, (_, name) in self._loose_legs.items()
                   if name and name[0] == 'p')

    def nodes_with_leg(self, leg):
        """Returns the two nodes that border on the given leg.
        """
        for u, v, ll in self.to_undirected(as_view=True).edges(data='leg'):
            if ll == leg:
                return u, v

    def contracted_nodes(self, A, B, simplify=True):
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
        T = C.contract(B, (list(C.connections(B)),) * 2)
        if simplify:
            T = T.simplify()
        result = TNS(tuple(T if n == A else n for n in self if n != B))

        if self.orthoCenter in (A, B):
            result._orthoCenter = T

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
        if not nx.algorithms.dag.is_directed_acyclic_graph(self):
            raise ValueError('Graph should be a directed acyclic graph')
        for n, d in self.degree():
            if d == 1:
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
        if self.orthoCenter != node:
            raise ValueError('node is not orthocenter')

        tns = TNS(T for T in self if T != node)
        Q, R = node.qr(leg)
        tns.add_nodes_from((Q, R))
        tns._orthoCenter = R
        tns.name_loose_edges_from([[ll, name] for ll, (_, name) in
                                   self._loose_legs.items()])
        if intermittent_renorm:
            R /= np.linalg.norm(R.ravel())
        return tns, (Q, R)

    def svd(self, node, leg, maxD=None):
        """Does an SVD on a node along the internal leg.

        The singular values are saved on the leg itself and also returned.
        """
        maxD = maxD if maxD else self.max_bondDimsension
        tns = TNS(T for T in self if T != node)
        U, S, V = node.svd(leg, compute_uv=True, maxD=maxD)
        tns.add_nodes_from((U, V))

        # TODO: Should do this in another way
        tns.edges[U, V, 0]['singular values'] = S

        tns.name_loose_edges_from([[ll, name] for ll, (_, name) in
                                   self._loose_legs.items()])
        return tns, U, S, V

    def swap(self, leg1, leg2, ortho_on=None, **kwargs):
        """Returns a network where leg1 and leg2 are swapped places

        At this moment leg1 and leg2 should be part of neighbouring tensors

        Args:
            leg1, leg2: the legs and flows to swap.
            Can be either loose_legs or edges of the network
        """
        ortho_on = leg1[0] if not ortho_on else ortho_on
        assert ortho_on in (leg1[0], leg2[0])
        As = []
        for ll, f in (leg1, leg2):
            if ll in self._loose_legs:
                As.append(self._loose_legs[ll][0])
            else:
                for X, Y, leg in self.edges(data='legs'):
                    if leg == ll:
                        As.append(Y if f else X)
                        break

        A, B = As
        cA, cB = [X.coupling_id(l[0]) for l, X in zip((leg1, leg2), (As))]
        assert A.coupling[cA[0]][cA[1]][1] == leg1[1]
        assert B.coupling[cB[0]][cB[1]][1] == leg2[1]

        if not A.connections(B):
            raise ValueError('leg1 and leg2 not elements of neighboring nodes')

        tns, T = self.contracted_nodes(A, B, simplify=False)
        assert len(T.coupling) == 2
        T = T.swap(leg1[0], leg2[0])
        assert len(T.coupling) == 2
        tns, U, S, V = tns.svd(T, T.internallegs[0], **kwargs)
        X = U if ortho_on in U.indexes else V
        assert ortho_on in X.indexes
        X @= S
        tns._orthoCenter = X
        return tns

    def reortho(self):
        """Reorthogonalizes the network
        """
        for tns, _, _ in self.depthFirst(loop=True):
            pass
        return tns

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

    def traverse_depthFirst(self, current_ortho=None, loop=False):
        current_ortho = self.sink if not current_ortho else current_ortho
        boundaries = set(b for b in self.boundaries())
        if current_ortho in boundaries:
            boundaries.remove(current_ortho)
            current = current_ortho
            G = self.to_undirected(as_view=True)
            while boundaries:
                shortest = min((nx.shortest_path(G, current, b) for b in
                                boundaries), key=len)
                current = shortest[-1]
                boundaries.remove(current)
                for A, B in zip(shortest, shortest[1:]):
                    yield A, B
            if loop:
                shortest = nx.shortest_path(G, current, current_ortho)
                for A, B in zip(shortest, shortest[1:]):
                    yield A, B

    def depthFirst(self, current_ortho=None, loop=False):
        assert nx.algorithms.dag.is_directed_acyclic_graph(self)
        current_ortho = self.sink if not current_ortho else current_ortho

        legs = [A.connections(B).pop() for A, B in
                self.traverse_depthFirst(current_ortho, loop)]
        leg_map = {}

        A = current_ortho
        tns = self
        for oleg in legs:
            leg = leg_map.get(oleg, oleg)
            nA, nB = tns.nodes_with_leg(leg)
            # A is the current ortho center
            assert A == nA or A == nB
            B = nB if nA == A else nA  # B is the other tensor than A
            tns, (Q, R) = tns.qr(A, leg)
            tns, A = tns.contracted_nodes(R, B)
            leg_map[oleg] = A.connections(Q).pop()
            yield tns, Q, A

    def two_rdms(self, current_ortho=None):
        rdms = {}
        pLegs = self.getPhysicalLegs()
        intermediate = {pl: {} for pl in pLegs}

        # Adjoint mapping for all physical legs
        padj_map = {pl: Leg(leg=pl) for pl in pLegs}
        # Adjoint mapping for virtual legs
        vadj_map = {}

        totals = len(self.orbitals) * (len(self.orbitals) - 1) // 2
        for _, Q, A in self.depthFirst(current_ortho):
            # print(f'{len(rdms)} / {totals}')
            nleg = A.connections(Q).pop()
            Qa = Q.adj(nleg)
            Qaset = set(Qa.indexes).difference(Q.indexes)
            assert len(Qaset) == 1
            vadj_map[nleg] = Qaset.pop()  # update adjoint mapping

            # Initiating of new intermediate result (only if Q is physical)
            pl = set(Q.indexes).intersection(pLegs)
            if pl:
                Qa2 = Qa.shallowcopy().swaplegs(padj_map)
                pl = pl.pop()
                intermediate[pl][nleg] = (Q @ Qa2).swap(vadj_map[nleg], pl)

            # Updating intermediate results with Q
            for pl, dic in intermediate.items():
                if pl in Q.indexes:
                    continue

                ic = set(Q.indexes).intersection(dic)
                if ic:
                    # An intermediate result has to be on exactly one side of Q
                    assert len(ic) == 1
                    ll = ic.pop()
                    Qa2 = Qa.shallowcopy().swaplegs({ll: vadj_map[ll]})
                    dic[nleg] = (dic[ll] @ Q) @ Qa2
                    assert len(dic[nleg].coupling) == 2

            # Forming finished rdms with A
            Aa = A.adj(leg=None).swaplegs(padj_map)
            # First case: A is a physical tensor
            pl = set(A.indexes).intersection(pLegs)
            if pl:
                pl = pl.pop()
                for pl2, dic in intermediate.items():
                    if pl2 == pl or (pl, pl2) in rdms or (pl2, pl) in rdms:
                        continue

                    ic = set(A.indexes).intersection(dic)
                    if ic:
                        # An intermediate has to be on exactly one side of A
                        assert len(ic) == 1
                        ll = ic.pop()
                        Aa2 = Aa.shallowcopy().swaplegs({ll: vadj_map[ll]})
                        rdms[(pl, pl2)] = (dic[ll] @ A) @ Aa2
                        assert len(rdms[(pl, pl2)].coupling) == 2
            else:
                # Second case: A is a branching tensor
                # No recombination needed
                pass
        for k in rdms:
            rdms[k].swap(k[0], padj_map[k[1]])

        assert len(rdms) == totals
        return rdms

    def mutualInfo(self, alphas=[1], current_ortho=None):
        """Order is p1, p2, ...
        """
        from sloth.utils import renyi_entropy
        pLegs = self.lexicographicalPsiteSort()
        rdms = self.two_rdms(current_ortho)
        svals = self.calculate_singular_values(current_ortho)
        pLsvals = {
            **{ll: r.svd(leg=r.internallegs[0])[1] for ll, r in rdms.items()},
            **{l: svals[l] for l in pLegs}
        }
        if not hasattr(alphas, '__iter__'):
            alphas = [alphas]
        pLentropies = {ll: np.array([renyi_entropy(v, α=a) for a in alphas])
                       for ll, v in pLsvals.items()}

        mutualInformation = np.zeros((len(pLegs), len(pLegs), len(alphas)))
        for ii, l1 in enumerate(pLegs):
            Si = pLentropies[l1]
            for jj, l2 in enumerate(pLegs[ii + 1:], start=ii + 1):
                key = (l1, l2) if (l1, l2) in pLentropies else (l2, l1)
                Sj = pLentropies[l2]
                Sij = pLentropies[key]
                I = 0.5 * (Si + Sj - Sij)
                mutualInformation[ii, jj, :] = I
                mutualInformation[jj, ii, :] = I

        mutualInformation = np.squeeze(mutualInformation)
        return mutualInformation

    def reorderPhysical(self, neworder, current_ortho=None, maxD=None):
        """Reordering the physical indices on the tensor network.

        At this moment only for MPSs.
        Args:
            neworder: List of legs for the new configuration topological
            ordered.
        """
        current_ortho = self.sink if not current_ortho else current_ortho
        if not self.isMPS:
            raise NotImplementedError('Only for MPS implemented')

        swapped = True
        # Bubble sort
        legs = [A.connections(B).pop() for A, B in
                self.traverse_depthFirst(current_ortho, True)]
        leg_map = {}

        A = current_ortho
        tns = self
        while swapped:
            swapped = False
            for oleg in legs:
                leg = leg_map.get(oleg, oleg)
                nA, nB = tns.nodes_with_leg(leg)
                # A is the current ortho center
                assert A == nA or A == nB
                B = nB if nA == A else nA  # B is the other tensor than A
                Al = set(A.indexes).intersection(neworder)
                Bl = set(B.indexes).intersection(neworder)
                assert len(Al) == 1
                assert len(Bl) == 1
                Al, Bl = Al.pop(), Bl.pop()
                if (neworder.index(Al) > neworder.index(Bl)) != \
                        (B in nx.ancestors(tns, A)):
                    # need a swap
                    tns = tns.swap((Al, True), (Bl, True), ortho_on=Bl,
                                   maxD=maxD)
                    Al, Bl = Bl, Al
                    swapped = True

                A = tns._loose_legs[Al][0]
                B = tns._loose_legs[Bl][0]
                leg = A.connections(B).pop()

                tns, (Q, R) = tns.qr(A, leg)
                tns, A = tns.contracted_nodes(R, B)
                leg_map[oleg] = A.connections(Q).pop()

        return tns

    def getPhysicalDistances(self, lexico=False):
        """Returns a matrix with the path lengths between all the physical
        legs.

        Returns:
            Distance between each two physical legs
        """
        if lexico:
            pLegs = self.lexicographicalPsiteSort()
        else:
            pLegs = self.topologicalPsiteSort()
        pSites = {N: set(pLegs).intersection(N.indexes).pop() for N in self
                  if set(pLegs).intersection(N.indexes)}
        D = np.zeros((len(pLegs), len(pLegs)), dtype=int)

        for N, dists in nx.all_pairs_shortest_path_length(
                self.to_undirected(as_view=True)):
            if N in pSites:
                ii = pLegs.index(pSites[N])
                for M, v in dists.items():
                    if M in pSites:
                        jj = pLegs.index(pSites[M])
                        D[ii, jj] = v
        return D

    def _write_h5_network(self, f, bleg, sites):
        """Helper function to write the network in the opened hdf5 file

        Args:
            f: the id of the root of the h5 file to write in
        Returns:
            sites: list with all the nodes in the used order
            bleg: list with all the virtual edges in the used order
        """
        pLegs = self.getPhysicalLegs()

        h5n = f.create_group('network')
        h5n.attrs['sites'] = [np.int32(len(sites))]
        h5n.attrs['psites'] = [np.int32(len(self.orbitals))]
        h5n['sweep'] = [np.int32(sites.index(A)) for A, _ in
                        self.traverse_depthFirst(loop=True)]
        h5n.attrs['sweeplength'] = [np.int32(len(h5n['sweep']))]
        h5n['sitetoorb'] = [
            np.int32(self._loose_legs[
                pLegs.intersection(A.indexes).pop()][1][1:])
            if pLegs.intersection(A.indexes) else -1 for A in sites]

        bonds = []
        for A in sites:
            Ai = sites.index(A)
            if len(list(self.predecessors(A))) == 0:
                bonds.append([-1, Ai])
                assert A.coupling[0][0][0].vacuum
            successors = set(self.successors(A))
            if successors:
                assert len(successors) == 1
                bonds.append([Ai, sites.index(successors.pop())])
                assert not A.coupling[0][2][0].vacuum
            else:
                assert A == self.sink
                bonds.append([Ai, -1])
                assert not A.coupling[0][2][0].vacuum
        h5n['bonds'] = np.array(bonds, dtype=np.int32).ravel()
        h5n.attrs['nr_bonds'] = [np.int32(len(bonds))]

    def _write_h5_symsec(self, h5b, A, bleg, ll):
        pLegs = self.getPhysicalLegs()
        bid = int(self._loose_legs[ll][1][1:]) if ll in pLegs \
            else bleg.index(ll)
        h5s = h5b.create_group(f'{"p" if ll in pLegs else "v"}_symsec_{bid}')
        bid += 2 * h5b.attrs['nr_bonds'].item() if ll in pLegs else 0
        h5s.attrs['bond'] = [np.int32(bid)]

        ci1, ci2 = A.coupling_id(ll)
        cii = A.indexes.index(ll)
        qnd = set((k[ci1][ci2], v.shape[cii]) for k, v in A.items())

        if ll not in pLegs:
            # Sort with respect to the keys
            qnd = sorted(qnd, key=lambda tup: tup[0][::-1])
        else:
            # Weird conventions in T3NS
            U1ids = sorted(A.getSymmIds('U(1)'), reverse=True)
            qnd = sorted(qnd, key=lambda tup: [tup[0][i] for i in U1ids])

        # Every key has only one entry with an unique dimension
        assert len(qnd) == len(set(q for q, _ in qnd))

        qn, di = [x for x in zip(*qnd)]
        h5s['irreps'] = np.array(qn, dtype=np.int32).flatten()
        h5s['dims'] = np.array(di, dtype=np.int32)
        h5s['fcidims'] = np.ones(len(qn), dtype=np.float64)
        h5s.attrs['totaldims'] = [np.int32(sum(h5s['dims']))]
        h5s.attrs['nrSecs'] = [np.int32(len(qn))]
        return qn, di

    def _write_h5_bookkeeper(self, f, sites, bleg):
        h5b = f.create_group('bookkeeper')
        h5b.attrs['nrSyms'] = [np.int32(len(self.sink.symmetries))]
        h5b.attrs['Max_symmetries'] = h5b.attrs['nrSyms']
        h5b.attrs['nr_bonds'] = f['network'].attrs['nr_bonds']
        h5b.attrs['psites'] = f['network'].attrs['psites']
        h5b.attrs['sgs'] = [np.int32(_SYMMETRIES.index(s))
                            for s in self.sink.symmetries]

        # Find target bond
        target_leg = set(ll for ll, f in self.sink.coupling[0] if not f).pop()
        id1, id2 = self.sink.coupling_id(target_leg)
        assert id1 == 0 and id2 == 2
        # filthy fix: but that is because it is ugly in T3NS-code
        target = [set(x).pop() if len(set(x)) == 1 else -max(set(x))
                  for x in zip(*[k[id1][id2] for k in self.sink])]
        assert len(target) == len(self.sink.symmetries)
        h5b.attrs['target_state'] = np.array(target, dtype=np.int32)

        # data for the symsecs
        qnsecs = {}
        for A in self:
            for ll in set(A.indexes).difference(qnsecs):
                # Writing the symsecs
                qnsecs[ll] = self._write_h5_symsec(h5b, A, bleg, ll)
        return qnsecs

    def _write_h5_T3NS(self, f, qnsecs, sites):
        h5t = f.create_group('T3NS')
        h5t.attrs['nrSites'] = [np.int32(len(sites))]

        def prefactor(key):
            return np.prod([np.sqrt(key[0][2][i] + 1) for i in SU2_ids])

        for ii, A in enumerate(sites):
            SU2_ids = [] if A == self.sink else A.getSymmIds('SU(2)')

            h5A = h5t.create_group(f'tensor_{ii}')
            h5A.attrs['nrblocks'] = [np.int32(len(A))]
            h5A.attrs['nrsites'] = [np.int32(len(A.coupling))]
            h5A.attrs['sites'] = [np.int32(ii)]
            h5b = h5A.create_group('block_0')
            h5b.attrs['nrBlocks'] = h5A.attrs['nrblocks']
            blocks = []
            for k, v in A.items():
                i1, i2, i3 = [qnsecs[ll][0].index(kk) for (ll, _), kk in
                              zip(A.coupling[0], k[0])]
                a, b, c = [len(qnsecs[ll][0]) for ll, _ in A.coupling[0]]
                qn = i1 + i2 * a + i3 * a * b
                assert v.ndim == 3
                # Fortran order
                v2 = (v / prefactor(k)).ravel(order='F')
                assert False not in [x == y[0] for x, y
                                     in zip(A.indexes, A.coupling[0])]
                assert [True, True, False] == [x for _, x in A.coupling[0]]
                blocks.append((qn, v.size, v2))
            blocks = sorted(blocks, key=lambda x: x[0])
            qn, sizes, blocks = list(zip(*blocks))
            h5b['tel'] = np.concatenate(blocks)
            h5b['beginblock'] = np.concatenate(([0], np.cumsum(sizes)))
            h5A['qnumbers'] = qn

    def _preprocess_h5_network(self):
        """Manipulates the current network such that the T3NS follows the
        convention as in the T3NS C-code (concerning coupling orders and so)
        """
        from copy import deepcopy
        if not nx.algorithms.dag.is_directed_acyclic_graph(self):
            raise ValueError('Graph should be a directed acyclic graph')
        if max(x for _, x in self.degree()) > 3:
            raise ValueError('Not a T3NS, nodes present with more than 3 legs')

        tns = deepcopy(self)

        # Check if the flow is correct for all the virtual legs,
        # otherwise swap its direction
        # TODO the swapping I should still implement
        assert set([tns.sink]).union(nx.ancestors(tns, tns.sink)) == set(tns)

        # order the sites
        sites = list(nx.topological_sort(tns))
        # order the virtual legs
        bleg = []
        lvedges = [l for l, (_, name) in tns._loose_legs.items() if not name]
        for A in sites:
            for ll, f in A.coupling[0]:
                if f and ll in lvedges:
                    bleg.append(ll)
                elif not f:
                    bleg.append(ll)

        # change the coupling order if needed, i.e.
        # The coupling should be IN, IN, OUT where the second IN is either
        # a physical leg or appears after the first leg in bleg and
        # OUT should be last in bleg
        #
        # The indexes should also be in this order!
        for A in sites:
            assert len(A.coupling) == 1
            # Sort legs
            lsrt = sorted(A.coupling[0], key=lambda x: bleg.index(x[0]) if
                          x[0] in bleg else -1)
            # Possibly physical legs are now in the first spot! should fix this
            if lsrt[0][0] not in bleg:
                lsrt[0], lsrt[1] = lsrt[1], lsrt[0]

            assert lsrt[0][0] in bleg and lsrt[2][0] in bleg
            assert [y for x, y in lsrt] == [True, True, False]
            permute = [A.coupling[0].index(x) for x in lsrt]
            A = A._swap0(0, permute)
            assert A.coupling[0] == tuple(lsrt)

            # Swapping the indexes of the tensor itself
            permuteid = [A.indexes.index(x) for x, y in A.coupling[0]]
            for k in A:
                A[k] = np.transpose(A[k], axes=permuteid)
            A._indexes = [x for x, y in A.coupling[0]]

        return tns, bleg, sites

    def write_h5(self, filename, oldfilename):
        """Writes to a new hdf5file.
        """
        import h5py
        tns, bleg, sites = self.reortho()._preprocess_h5_network()

        fo = h5py.File(oldfilename, 'r')
        fn = h5py.File(filename, "w")
        try:
            fo.copy('hamiltonian', fn)  # copying hamiltonian

            tns._write_h5_network(fn, bleg, sites)  # write the network
            qnsecs = tns._write_h5_bookkeeper(fn, sites, bleg)  # write bookie
            tns._write_h5_T3NS(fn, qnsecs, sites)  # write wave function
        finally:
            fo.close()
            fn.close()

    def plotEntanglement(self, mutualInfo=None, vmin=0, vmax=None,
                         cmap='tab20b', **kwargs):
        """Makes a plot of the entanglement in each bond and the mutual
        information in topological order
        """
        import matplotlib.pyplot as plt
        from sloth.utils import renyi_entropy

        nodes = list(nx.topological_sort(self))
        pLegs = self.getPhysicalLegs()
        X = [pLegs.intersection(N.indexes) for N in nodes]
        pLegs = [x.pop() for x in X if x]
        topo_edges = [A.connections(B).pop() for A, B in zip(nodes, nodes[1:])]

        if mutualInfo is None:
            mutualInfo = self.mutualInfo(**kwargs)
        # sort to topo order
        lexicoL = self.lexicographicalPsiteSort()
        swap = [lexicoL.index(ll) for ll in pLegs]
        mutualInfo = mutualInfo[swap, ...][:, swap, ...]

        svals = self.calculate_singular_values()

        x = np.arange(start=0.00, stop=1.01, step=0.01)
        sv = np.array([[renyi_entropy(svals[k], a) for k in topo_edges]
                       for a in x])

        keys = {
            'origin': 'lower',
            'aspect': 'auto',
            'cmap': cmap,
            'vmin': vmin,
            'vmax': vmax
        }

        fig, axs = plt.subplots(ncols=2)

        mat = axs[0].matshow(sv, **keys)
        axs[0].set_ylabel(r'Rényi $\alpha$-value')
        axs[0].set_yticklabels([f"{yy / 100.:.1f}"
                                for yy in axs[0].get_yticks()])
        axs[0].xaxis.tick_bottom()
        axs[0].set_xlabel('bond')
        fig.colorbar(mat, ax=axs[0], pad=0.02, fraction=0.1)
        mat = axs[1].matshow(mutualInfo, aspect='auto', cmap='Greys')
        fig.colorbar(mat, ax=axs[1], pad=0.02, fraction=0.1)
        return fig, axs


def read_h5(filename):
    """This reads a hdf5 file.

    Returns the network.
    """
    import h5py
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
    h5file.close()
    tns._orthoCenter = tns.sink

    assert nx.algorithms.dag.is_directed_acyclic_graph(tns)
    return tns

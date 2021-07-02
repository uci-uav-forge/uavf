from networkx.utils.decorators import preserve_random_state
from numpy.random.mtrand import vonmises
from p2 import polygon
import networkx as nx
import numpy as np
import copy
from matplotlib import pyplot as plt
import enum

class Event(enum.Enum):
    CLOSE=1
    OPEN=2
    SPLIT=3
    MERGE=4
    INFLECTION=5
    INTERSECT=6

def line_sweep(G: nx.DiGraph, theta: float, posattr: str = 'points'):
    # sorted node list becomes our list of events
    H = rotate_graph(G, theta, posattr=posattr)
    # sort left to right and store sorted node list in L.
    def left_to_right(n): return H.nodes[n][posattr][0]
    L = sorted(list(H.nodes), key=left_to_right)
    # sweep left-to-right
    crits, cells = [], []
    for v in L:
        etype = check_lu(H, v)
        if etype == Event.SPLIT or etype == Event.MERGE:
            splitmerge_points(H, v)
        # crits
        if etype in (Event.OPEN, Event.SPLIT, Event.MERGE, Event.CLOSE):
            crits.append(v)
    for c in crits:
        cells = append_cell(H, c, cells)
    build_reebgraph(H, cells)

def build_reebgraph(H: nx.DiGraph, cells: list) -> None:
    edges = []
    for i, a in enumerate(cells):
        for j, b in enumerate(cells):
            union = a & b
            if len(union) == 2:
                edges.append(tuple(union))



def rcross(H: nx.DiGraph, u:int, v:int, w:int) -> float:
    '''compute the vector cross product of edges `u`,`v` and `v`,`w` on `H`.
    This one is more expensive than qcross but returns a numerical value.

    Parameters
    ----------
    H : nx.DiGraph
        Graph
    u : int
        idx of node
    v : int
        idx of node
    w : int
        idx of node

    Returns
    -------
    float
        cross product
    '''
    a = H.nodes[v]['points'] - H.nodes[u]['points']
    b = H.nodes[w]['points'] - H.nodes[v]['points']
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return a[0] * b[1] - a[1] * b[0]

def append_cell(H: nx.DiGraph, v: int, cells: list) -> list:
    '''[summary]

    Parameters
    ----------
    H : nx.DiGraph
        graph on which to append cells
    v : int
        vertex from which to start
    cells : list
        list of cells.

    Returns
    -------
    list
        new list of cells
    '''
    prev_node = v
    for path_start in H.adj[v]:
        path_end = False
        path = []
        node = path_start
        while not path_end:
            cvals = []
            for candidate in H.adj[node]:
                if candidate != prev_node:
                    cval = rcross(H, prev_node, node, candidate)
                    cvals.append( (candidate, cval) )
            def cvalsort(c): return c[1]
            cvals.sort(key=cvalsort)
            # get most clockwise cval
            best = cvals[0][0]
            prev_node = node
            node = best
            path.append(best)
            if best == path_start:
                path_end = True
        if set(path) not in cells:
            cells.append(set(path))
    return cells

def splitmerge_points(H: nx.DiGraph, v: int):
    a, b = get_intersects(H, v)
    if a:
        ai = len(H.nodes()) + 1
        H.add_node(ai, points=a['pt'])
        # add new edge to H
        H.add_edge(ai, v, weight=3) # close
        H.add_edge(v, ai, weight=4) # open
        # split the edge with which there was a collision in half
        H.add_edge(a['e1'], ai, weight=a['weight'])
        H.add_edge(ai, a['e2'], weight=a['weight'])
        H.remove_edge(a['e1'], a['e2'])
    if b:
        bi = len(H.nodes()) + 1
        H.add_node(bi, points=b['pt'])
        # add new edge to H
        H.add_edge(v, bi, weight=3) # open
        H.add_edge(bi, v, weight=4) # close
        # split the edge with which there was a collision in half
        H.add_edge(b['e1'], bi,  weight=b['weight'])
        H.add_edge(bi, b['e2'],  weight=b['weight'])
        H.remove_edge(b['e1'], b['e2'])

def get_intersects(H: nx.DiGraph, v: int) -> tuple:
    '''Check for intersects a vertical line passing through v on the graph H.
    Think of it like this: We basically draw a line upwards from v until we 
    reach a line on H; then we register a collision, which contains the origin 
    vertex of the collision in 'vx', the point of the collision in 'pt', the edge
    with which the line collided in 'edge', and the weight of that edge in 'weight'.

    This function returns a maximum of two collisions, one above and one below, 
    which correspond to the closest edge with which the line starting at `v` 
    collided. If there is no such edge, it returns `None`, otherwise returns a
    dict with information about the collision specified above.

    Parameters
    ----------
    H : nx.DiGraph
        graph
    v : int
        event idx

    Returns
    -------
    tuple
        collisions above or/and below v on edges of H
    '''
    # list of all collisions with vertex v on the vertical
    collisions = []
    # check each edge
    for edge in H.edges:
        # check whether there could be a collision
        if check_edge(H, v, edge):
            e1, e2 = edge
            p1, p2, pv = H.nodes[e1]['points'], H.nodes[e2]['points'], H.nodes[v]['points']
            ipt = intersect_vertical_line(p1, p2, pv)
            # store attrs of collision
            collisions.append({
                'vx' : v,
                'pt' : ipt,
                'e1' : e1,
                'e2' : e2,
                'weight' : H[e1][e2]['weight']
            })
    above, below = None, None
    y = H.nodes[v]['points'][1]
    # the difference between the collision point and the vertex point
    def ydiff(c): return c['pt'][1] - y
    if collisions:
        # the minimum collision point above `v`
        above = min([c for c in collisions if c['pt'][1] > y], key=ydiff, default=None)
        # the maximum collision point below `v`
        below = max([c for c in collisions if c['pt'][1] < y], key=ydiff, default=None)
    return above, below

def intersect_vertical_line(p1: np.ndarray, p2: np.ndarray, pv: np.ndarray) -> np.ndarray:
    m = (pv[0] - p1[0]) / (p2[0] - p1[0])
    # y = mx + b
    py = m * (p2[1] - p1[1]) + p1[1]
    return np.array([pv[0], py])

def check_edge(H: nx.DiGraph, v: int, edge: tuple) -> bool:
    '''Check whether an edge [edge=(e1, e2)] on H intersects with a straight vertical line
    at `v`

    Parameters
    ----------
    H : nx.DiGraph
        graph
    v : int
        vertex idx
    edge : tuple of (int, int)
        edge idxs

    Returns
    -------
    bool
        Whether the x coordinates of the point `v` are beteen the edge x-coords
    '''
    e1, e2 = edge
    p1, p2, pv = H.nodes[e1]['points'], H.nodes[e2]['points'], H.nodes[v]['points']
    if p1[0] > pv[0] and p2[0] < pv[0]: return True
    elif p1[0] < pv[0] and p2[0] > pv[0]: return True
    else: return False

def check_lu(H: nx.DiGraph, v:int) -> int:
    l, u, above = lower_upper(H, v)
    lpoint = H.nodes[l]['points'] # lower point
    upoint = H.nodes[u]['points'] # upper point
    vpoint = H.nodes[v]['points'] # vertex point

    event = None
    # both points on the right of v
    if lpoint[0] > vpoint[0] and upoint[0] > vpoint[0]:
        if above: event = Event.OPEN
        else: event = Event.SPLIT
    # both points on the left of v
    elif lpoint[0] < vpoint[0] and upoint[0] < vpoint[0]:
        if above: event = Event.CLOSE
        else: event = Event.MERGE
    # lower right, upper left
    elif lpoint[0] > vpoint[0] and upoint[0] < vpoint[0]:
        event = Event.INFLECTION
    # lower left, upper right
    elif lpoint[0] < vpoint[0] and upoint[0] > vpoint[0]:
        event = Event.INFLECTION

    if event is None:
        raise(ValueError('Event was not categorized correctly!'))
    return event
    
def lower_upper(H: nx.DiGraph, v: int):
    '''Check the neighbors of `v` to determine which is the lower neighbor 
    and which is the upper neighbor. If the predecessor is the upper neighbor,
    also returns True 

    Parameters
    ----------
    H : nx.DiGraph
        must contain 'weight' as a key.
    v : int
        The vertex to check

    Returns
    -------
    tuple of (int, int, bool)
        lower vertex, upper vertex, above
    '''

    # filter only for neighboring nodes which were not constructed by a split
    # event i.e. have weight 3...
    vA, vB = None, None
    for p in H.predecessors(v):
        w = H[p][v]['weight']
        if w == 1 or w == 2: 
            vA = p
    for s in H.successors(v):
        w = H[v][s]['weight']
        if w == 1 or w == 2:
            vB = s
    if vA is None or vB is None:
        raise(ValueError('No valid predecessors or successors to node {}'.format(v)))
    # compute cross product. if qcross is true (i.e. positive cross product), then 
    # the vertex vA is "above" the vertex vB. ["above" here means "above" in a 
    # topological sense; it's not necessarily above in the cartesian sense.]
    # 
    # Note that when "above" is true, vB and vA are flipped (i.e. the successor
    # is stored into lower, and the predecessor is stored into upper.
    above = qcross(H, vA, v, vB)
    if above: 
        lower, upper = vB, vA
    else: 
        lower, upper = vA, vB
    return lower, upper, above

def qcross(H: nx.DiGraph, vA: int, v: int, vB: int) -> bool:
    '''compute a quick cross product over nodes `vA`, `v`, `vB` on `H`.'''
    # find vectors with tails at `v` and pointing to vA, vB
    p1 = H.nodes[vA]['points']
    p2 = H.nodes[vB]['points']
    pv = H.nodes[v]['points']

    a = pv - p1
    b = pv - p2
    # this is roughly a measure of the topological orientation of vA, vB
    if a[0] * b[1] - b[0] * a[1] >= 0: return True
    else: return False

def rotate_graph(G: nx.DiGraph, theta: float, posattr: str = 'points') -> nx.DiGraph:
    '''Rotate a graph `G`. This makes a copy of `G` and returns it, leaving `G` untouched.

    Parameters
    ----------
    G : nx.DiGraph
        input Graph
    theta : float
        rotation angle, radians
    posattr : str, optional
        which key the points are stored under, by default 'points'

    Returns
    -------
    nx.DiGraph
        the new rotated graph
    '''
    H = copy.copy(G)
    rotmatr = make_rot_matrix(theta)
    for node in G.nodes:
        original = G.nodes[node]['points']
        rotated = original @ rotmatr
        H.nodes[node]['points'] = rotated
    return H

def make_rot_matrix(theta: float) -> np.ndarray:
    '''Create a rotation matrix

    Parameters
    ----------
    theta : float
        Angle

    Returns
    -------
    np.ndarray
        2x2 Rotation Matrix created from Angle.
    '''
    rotmatr = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return rotmatr


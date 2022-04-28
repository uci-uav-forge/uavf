from networkx.algorithms.traversal.edgedfs import edge_dfs
import numpy as np
import networkx as nx
from p2 import polygon
import matplotlib.pyplot as plt


def get_cell_closed(G: nx.DiGraph, cell):
    cycs = nx.simple_cycles(G.subgraph(cell))
    cycle = max(cycs, key=lambda e: len(e))
    edges = [(cycle[i], cycle[i + 1]) for i, _ in enumerate(cycle[:-1])] + [
        (cycle[-1], cycle[0])
    ]
    return G.edge_subgraph(edges).copy()


def make_grid(C: nx.DiGraph, gridstep=0.1):
    points = np.empty((len(C.nodes), 2))
    for i, n in enumerate(C.nodes):
        points[i] = C.nodes[n]["points"]
    # get min, max
    maxx, maxy = np.max(points[:, 0]), np.max(points[:, 1])
    minx, miny = np.min(points[:, 0]), np.min(points[:, 1])
    # grid
    xs = np.arange(minx + gridstep / 2, maxx - gridstep / 2, gridstep)
    ys = np.arange(miny + gridstep / 2, maxy - gridstep / 2, gridstep)
    grid = np.empty((xs.shape[0], ys.shape[0], 2))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            print(x, y)
            grid[i, j] = np.array([x, y])
    print(grid)


def sweep_cell(G: nx.DiGraph, cell):
    C = get_cell_closed(G, cell)
    points = make_grid(C)
    return "test"


def make_path(G: nx.DiGraph, S: nx.DiGraph):
    visited = []
    for e1, e2 in nx.eulerian_circuit(nx.eulerize(S)):
        if S.nodes[e1]["cell"] is not None:
            cell = set(S.nodes[e1]["cell"])
            if cell not in visited:
                path = sweep_cell(G, cell)
                print(path)
                visited.append(cell)
    return S
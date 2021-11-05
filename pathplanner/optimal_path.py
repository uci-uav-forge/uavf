from p2.polygon import polygon, draw_G, cluster_points
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from p2.bcd import line_sweep
from scipy.spatial import Delaunay, delaunay_plot_2d


def get_points(G: nx.DiGraph(), cell):
    return np.array([G.nodes[v]["points"] for v in cell])


def triangle_coll(tri, point):
    # calculate a triangle collision
    pass


if __name__ == "__main__":
    pts = cluster_points()
    # create grid from points
    xmin, xmax, ymin, ymax = (
        pts[:, 0].min(),
        pts[:, 0].max(),
        pts[:, 1].min(),
        pts[:, 1].max(),
    )
    gridpts = 115j
    xy = np.mgrid[xmin:xmax:gridpts, ymin:ymax:gridpts].reshape(2, -1).T

    G = polygon(pts, holes=0)

    H, cells = line_sweep(G, theta=0)
    for c in cells:
        # get points
        cell = get_points(H, c)
        # make delaunay triangulation
        dt = Delaunay(cell)
        # calculate collisions on xy
        delaunay_plot_2d(dt)

    fig, ax = plt.subplots()
    draw_G(H, ax)
    ax.scatter(xy[:, 0], xy[:, 1], marker=".")
    plt.show()

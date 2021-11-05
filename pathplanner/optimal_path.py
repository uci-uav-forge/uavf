from p2.polygon import polygon, draw_G, cluster_points
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from p2.bcd import line_sweep
from scipy.spatial import Delaunay, delaunay_plot_2d


def get_points(G: nx.DiGraph(), cell):
    return np.array([G.nodes[v]["points"] for v in cell])

def tri_sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

def triangle_coll(point, v1, v2, v3):
    d1 = tri_sign(point, v1, v2)
    d2 = tri_sign(point, v2, v3)
    d3 = tri_sign(point, v3, v1)
     
    negative = d1 < 0 or d2 < 0 or d3 < 0
    positive = d1 > 0 or d2 > 0 or d3 > 0
    return not (negative and positive)

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

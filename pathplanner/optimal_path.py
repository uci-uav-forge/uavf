import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from p2 import polygon, bdc

if __name__ == "__main__":
    pts = polygon.beta_clusters()
    # create grid from points
    G, _, _ = polygon.polygon(pts, holes=4)
    J, R = bdc.line_sweep(G)
    W_disc = bdc.discretize_entire(J, R, gridsz=0.05)
    
    discrete_points = bdc.get_points_array(W_disc)
    

    


    fig, ax = plt.subplots()
    polygon.draw_G(W_disc, ax, node_text=False)
    # ax.scatter(xy[:, 0], xy[:, 1], marker=".")
    plt.show()

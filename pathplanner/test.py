from p2 import polygon, bcd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

if __name__ == '__main__':
   # generate a distribution of points in xy
   points = np.random.beta(4, 4, size=(30,2))
   # points = polygon.cluster_points(no_clusters=5, cluster_n=12, cluster_size=1, cluster_dist=1)
   # create a polygon and store it into G
   G = polygon.polygon(points, holes=3, removals=130)
   # sweep a line through the polygon and store 
   # J has the line sweep, R has the reeb graph,
   # H has the rotated line sweep

   ax = plt.axes()
   J, R, H, S = bcd.line_sweep(G, theta=bcd.degree2rad(0))
   # plot stuff
   # fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(7.5, 7.5), tight_layout=True)
   # for i, a in np.ndenumerate(ax):
   #    a.set_aspect('equal')
   #    a.tick_params(axis="both",which="both",bottom=False,left=False,labelbottom=False,labelleft=False)
   # ax[0,0].set_title('Point Distribution')
   # ax[0,1].set_title('Non-Convex Boundary')
   # ax[1,0].set_title('BDC in Rotated Coord System')
   # ax[1,1].set_title('BDC with Reeb Graph')
   # # first, points only
   # ax[0,0].scatter(points[:,0], points[:,1], marker='.', color='k')
   # # then graph formed from points by polygon algorithm
   # polygon.draw_G(G, ax[0,1], arrows=True)
   # # perform the line sweep
   # J, R, H = bcd.line_sweep(G, 0.5)
   # polygon.draw_G(H, ax[1,0], node_text=False)
   # polygon.draw_G(J, ax[1,1], ecolor='lightgray')
   # polygon.draw_G(R, ax[1,1], posattr='centroid', style='--')
   # plt.show()
   
   
   polygon.draw_G(J, ax, ecolor='gray',style='-', node_text=False, draw_nodes=False)
   polygon.draw_G(S, ax, ecolorattr='dist', node_text=False, arrows=True, style=':')
   ax.set_aspect('equal')
   for k1, v1 in S.nodes.items():
      print(k1)
      for k2, v2 in v1.items():
         print('\t', k2, ':', v2)
   plt.show()
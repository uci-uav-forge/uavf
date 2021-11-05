from p2 import polygon, bcd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def generate_points(n_clusters, n_points):
   csize = (int(n_points/n_clusters))
   points = np.empty( (csize*n_clusters, 2) )
   offset = np.zeros((1,2))
   for i in range(n_clusters):
      offset += np.random.normal(0, .4, (1, 2))
      dist = np.random.beta(4,4,size=(csize, 2)) + offset.repeat(csize, axis=0)
      points[i*csize:(i+1)*csize] = dist
   return points
      



if __name__ == '__main__':
   seed = np.random.randint(0,1e6)
   np.random.seed(seed)
   print(seed)
   # generate a distribution of points in xy
   points = generate_points(1, 20)

   G = polygon.polygon(points, holes=1, removals=13)
   ax = plt.axes()
   J, R, H, S, U = bcd.line_sweep(G, theta=bcd.degree2rad(10))   
   polygon.draw_G(J, ax, ecolor='gray',style='-', node_text=False, draw_nodes=False)
   polygon.draw_G(U, ax, ecolor='k')
   polygon.draw_G(S, ax, ecolor='r', node_text=False, arrows=True, style=':')
   ax.set_aspect('equal')
   # for k1, v1 in S.nodes.items():
   #    print(k1)
   #    for k2, v2 in v1.items():
   #       print('\t', k2, ':', v2)
   plt.show()
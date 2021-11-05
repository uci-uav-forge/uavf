from p2 import polygon
import numpy as np
from matplotlib import pyplot as plt
from p2 import bcd
from p2 import path
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
   points = generate_points(2, 17)
   J, R, H, S, U = bcd.line_sweep(polygon.polygon(points, holes=1, removals=10), theta=bcd.degree2rad(10))   
   fig, ax = plt.subplots()
   polygon.draw_G(J, ax=ax, ecolor='#ccc')
   polygon.draw_G(U, ax=ax)
   plt.show()
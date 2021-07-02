from p2 import polygon, bcd
import matplotlib.pyplot as plt
if __name__ == '__main__':
   G = polygon.polygon(polygon.cluster_points(), holes=4, removals=40)
   fig, ax = plt.subplots(ncols=2)
   polygon.draw_G(G, ax[0])
   bcd.line_sweep(G, 0.4)
   polygon.draw_G(G, ax[1])
   plt.show()

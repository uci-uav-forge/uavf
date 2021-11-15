from sklearn.cluster import KMeans
import cv2
from argparse import ArgumentParser
import hypertools as hyp
import matplotlib.pyplot as plt
import numpy as np


def read_image(fname, n_colors=4):
    raw = cv2.imread(fname)
    w, h = raw.shape[1], raw.shape[0]
    # LAB is more perceptually uniform so let's find blobs there
    raw = cv2.cvtColor(raw, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(raw)
    shaped = raw.reshape(w * h, 3)
    # hyp.plot(a, ".", reduce="TSNE", ndims=2, color="r", title="a")
    # hyp.plot(b, ".", reduce="TSNE", ndims=2, color="b", title="b")
    # plt.show()
    # )
    km = KMeans(n_colors).fit(shaped)
    dom_colors = km.cluster_centers_
    # get new image with 100px x 100px tiles
    cimg = np.empty((dom_colors.shape[0] * 100, 100, 3), dtype=np.uint8)
    for c in range(dom_colors.shape[0]):
        px = dom_colors[c, None]
        # set a tile
        cimg[c * 100 : (c + 1) * 100, :, :] = np.broadcast_to(px, (100, 100, 3))

    cimg_readable = cv2.cvtColor(cimg, cv2.COLOR_LAB2RGB)
    plt.imshow(cimg_readable)
    plt.show()


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("-i", required=True, type=str, help="Image File to Read")
    opts = ap.parse_args()

    read_image(opts.i)

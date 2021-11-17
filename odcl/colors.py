import sklearn.cluster as cluster
import cv2
import numpy as np


def make_colors_plot(ax, rgb_colors):
    """Make a plot on the Axes `ax` that shows the colors in the array `rgb_colors`

    Parameters
    ----------
    ax : matplotlib.Axes
        the axes onto which to draw the plot
    rgb_colors : np.ndarray
        shape (n_colors, 3) array of RGB colors

    Returns
    -------
    matplotlib.Axes
        updated axes object. It's not necessary to store this as it does not copy
    """
    cimg = np.empty((rgb_colors.shape[0] * 100, 100, 3), dtype=np.uint8)
    # go through colors
    for c in range(rgb_colors.shape[0]):
        px = rgb_colors[c, None]
        # set a tile
        cimg[c * 100 : (c + 1) * 100, :, :] = np.broadcast_to(px, (100, 100, 3))
    ax.imshow(cimg)
    return ax


def get_colors_kmeans(
    raw,
    resize_raw=1.0,
    n_colors=4,
    sample_pxs=5000,
    alpha=0.7,
    d=7,
    sigma1=15,
    sigma2=15,
    gkernelsize=(7, 7),
):
    """Get colors of an input image with k-means algorithm.

    Parameters
    ----------
    raw : np.ndarray (h, w, 3)
        Array representing RGB image. It is (h, w, 3) where (h, w) are the image height,
        width in pixels. You can use `cv2.imread(filename)` to get a suitable input
    resize_raw : float, optional
        amount to resize the image. this should be a float between 1 and 0; a value of 0.5
        will make an image which is a quarter (half width, height) as the original image
        this is because bilinear smoothing can be slow on very large images, so use this
        to speed up the bilinar filtering step, by default 1
    n_colors : int, optional
        no of colors to extract from the raw image, by default 4
    sample_pxs : int, optional
        number of pixels to sample. This is another scaling step used primarily to scale
        kmeans. The algorithm will sample this number of pixels over the image uniformly
        before computing clusters. More pixels sampled causes kmeans to take longer, but
        will generate more accurate clusters; accuracy will fall off significantly if this
        is set too low, by default 5000
    alpha: float, optional
        pre-smoothing coefficient. 0 means no pre-smoothing, 1 means entire image is pre-
        smoothed. Values in between are combinations of the smoothed and raw image.
    d : int, optional
        bilinear filter scale, by default 7
    sigma1 : int, optional
        bilinear filter scale sigma 1, by default 15
    sigma2 : int, optional
        bilinear filter scale sigma 2, by default 15
    gkernelsize : tuple, optional
        gaussian blur kernel size, by default (7, 7)

    Returns
    -------
    np.ndarray
        shape (n_colors, 3) array of rgb colors
    """
    # resize image
    rsz_w, rsz_h = int(resize_raw * raw.shape[1]), int(resize_raw * raw.shape[0])
    raw = cv2.resize(raw, (rsz_w, rsz_h))
    # LAB is more perceptually uniform so let's find blobs there
    raw = cv2.cvtColor(raw, cv2.COLOR_RGB2LAB)
    # bilateral filter -- make flat areas smoother
    bl = cv2.bilateralFilter(raw, d, sigma1, sigma2)
    gb = cv2.GaussianBlur(bl, gkernelsize, 0, 0)
    smoothed = np.minimum(alpha * gb + (1 - alpha) * raw, 255).astype("uint8")
    # all pixels
    all_pxs = smoothed.reshape(rsz_w * rsz_h, 3)
    # sampled pixels
    sam_pxs_i = np.random.choice(
        list(range(all_pxs.shape[0])), replace=False, size=sample_pxs
    )
    sam_pxs = all_pxs[sam_pxs_i, :]
    km = cluster.KMeans(n_colors).fit(sam_pxs)
    dom_colors = km.cluster_centers_
    return dom_colors


if __name__ == "__main__":
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt

    ap = ArgumentParser()
    ap.add_argument("-i", required=True, type=str, help="Image File to Read")
    opts = ap.parse_args()

    raw_image = cv2.imread(opts.i)
    colors = get_colors_kmeans(raw_image, resize_raw=2)
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(raw_image)
    make_colors_plot(ax[1], colors)
    plt.show()

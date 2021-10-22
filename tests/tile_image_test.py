from odcl import tile_image
import numpy as np

# will yield 4x4 image after tiling
def get_blank_16imgs():
    for i in (1792, 1793, 1795):
        for j in (1792, 1793, 1795):
            yield np.zeros((i, j, 3))


def get_blank_small_imgs():
    for i, j in ((100, 100), (447, 447)):
        yield np.zeros((i, j, 3))


def test_tile_image_size():
    for img in get_blank_16imgs():
        tiles = tile_image.tile_image(img, 448)
        assert len(tiles.keys()) == 16
    for img in get_blank_small_imgs():
        tiles = tile_image.tile_image(img, 448)
        assert len(tiles.keys()) == 0

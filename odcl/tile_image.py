import math


def tile_image(cv2imgarr, size):
    h, w = cv2imgarr.shape[0], cv2imgarr.shape[1]
    htiles = h // size
    wtiles = w // size

    # remaining tiles get halved and added like, e.g.
    # ┌───────────────┐
    # │               │
    # │ ┌─────┬─────┐ │
    # │ │     │     │ │
    # │ ├─────┼─────┤ │
    # │ │     │     │ │
    # │ └─────┴─────┘ │
    # │               │
    # └───────────────┘
    # so that the tiling is "centered" in the image.
    # we are throwing away data a little bit, but not
    # much (448 px for an efficientdet v2), and doing
    # inference on a partial area is probably no good

    hremain2 = math.floor(h % size / 2)
    wremain2 = math.floor(w % size / 2)

    imagetiled = {}

    for i in range(wtiles):
        for j in range(htiles):
            hlower = size * j + hremain2
            hupper = size * (j + 1) + hremain2
            wlower = size * i + wremain2
            wupper = size * (i + 1) + wremain2
            imagetiled[(i, j)] = cv2imgarr[hlower:hupper, wlower:wupper, :]

    return imagetiled

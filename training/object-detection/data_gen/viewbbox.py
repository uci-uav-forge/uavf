import cv2, os, argparse, math
from PIL import Image
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
	'-d',
	type = str,
	help = 'The dataset directory.',
	required = True
)
parser.add_argument(
    '-r',
    type=int,
    help='The number of rows',
    default=3,
    required=False,
)
parser.add_argument(
    '-c',
    type=int,
    help='No of columns',
    default=4,
    required=False
)
parser.add_argument(
    '-sz',
    type=int,
    help='resize multiple. 2x reduces image size by 2.',
    default=1.5,
    required=False
)
parser.add_argument(
    '-w',
    type=int,
    help='Width of bbox to draw',
    default=2,
    required=False
)
opt = parser.parse_args()


IMG_DIR = opt.d + 'images'
LAB_DIR = opt.d + 'labels'
# rows and cols of image to view
nrows = opt.r
ncols = opt.c
imgs = []
for img in os.listdir(IMG_DIR):
    imgs.append(
        (IMG_DIR + '/' + img,
        LAB_DIR + '/' + img[:-4] + '.txt')
    )

image = cv2.imread(imgs[0][0])
imagew = image.shape[0] * opt.r
imageh = image.shape[1] * opt.c
image = np.empty((imagew, imageh, 3), dtype=np.uint8)
classes = {}

imgsel = np.random.choice(len(imgs), nrows*ncols, replace=False)
for i in range(nrows):
    for j in range(ncols):
        # ipath, lpath = imgs[random.randint(0, len(imgs)-1)]
        ipath, lpath = imgs[imgsel[i*ncols+j]]

        img = cv2.imread(ipath)
        dh, dw, _ = img.shape
        fl = open(lpath, 'r')
        data = fl.readlines()
        fl.close()
        for dt in data:
            # Split string to float
            objclass, x, y, w, h = map(float, dt.split(' '))
            if objclass not in classes:
                classes[objclass] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

            l = int((x - w / 2) * dw)
            r = int((x + w / 2) * dw)
            t = int((y - h / 2) * dh)
            b = int((y + h / 2) * dh)
            if l < 0:
                l = 0
            if r > dw - 1:
                r = dw - 1
            if t < 0:
                t = 0
            if b > dh - 1:
                b = dh - 1
            alpha = 0.2
            cv2.rectangle(img, (l, t), (r, b), classes[objclass], opt.w)

        i1 = i * img.shape[0]
        i2 = i * img.shape[0] + img.shape[0]
        j1 = j * img.shape[1]
        j2 = j * img.shape[1] + img.shape[1]
        image[i1:i2, j1:j2] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
image = Image.fromarray(image)
image = image.resize(
    (math.ceil(imageh/opt.sz), math.ceil(imagew/opt.sz))
)
image.show()
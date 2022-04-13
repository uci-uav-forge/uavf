import cv2
import numpy as np

class Color:
    def __init__(self):
        pass
        
    #image processing methods
    def kmeans(self, img, k = 3):
        pixels = img.reshape(-1,3)
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.95)
        retval, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape((img.shape))
        return segmented_image, centers
    def adjust_gamma(self, img, gamma):
        invGamma = 1.0 / gamma
        table = np.array([((i/255.0) ** invGamma) * 255 for i in np.arange(0,256)]).astype('uint8')
        return cv2.LUT(img, table)

    def target_segmentation(self, img):
        lab = self.adjust_gamma(img, 0.5)
        lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)
        h,w,_ = lab.shape
        yGrid, xGrid = np.float32(np.mgrid[0:h,0:w])
        
        centerY = h/2
        centerX = w/2

        centerGrid = np.power( yGrid - centerY, 2 ) + np.power(xGrid - centerX, 2)
        modified_img = np.dstack( (lab, 0.2*centerGrid) )

        dataShape = modified_img.shape[2]

        data = modified_img.reshape(-1,dataShape)
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.95)

        #kmeans = 2
        k = 2
        _, labels, centers = cv2.kmeans( data, k , None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        shape_color = centers[np.argmin(centers[:,-1])][:3]

        off_img = lab[h//4:h-h//4, w//4:w-w//4]
        kmeans_img, kmeans_centers = self.kmeans(off_img,k=2)
        
        shapeidx = -1
        letteridx = -1
        distance1 = np.linalg.norm(kmeans_centers[0] - shape_color)
        distance2 = np.linalg.norm(kmeans_centers[1] - shape_color)

        if distance1 < distance2:
            shapeidx = 0
            letteridx = 1
        else:
            letteridx = 0
            shapeidx = 1

        kmeans_img = cv2.cvtColor( np.uint8(kmeans_img), cv2.COLOR_LAB2BGR)

        letter_mask = np.zeros(shape=(h,w,3), dtype=np.uint8)
        letter_mask[h//4:h-h//4, w//4:w-w//4] = kmeans_img

        shapeColor = cv2.cvtColor(np.uint8([[ kmeans_centers[shapeidx] ]]), cv2.COLOR_LAB2BGR)[0][0]
        letterColor = cv2.cvtColor(np.uint8([[ kmeans_centers[letteridx] ]]), cv2.COLOR_LAB2BGR)[0][0]
        letter_mask[ (letter_mask==shapeColor).all(axis=2) ] = [ 0,0,0 ]
        letter_mask = cv2.cvtColor(letter_mask, cv2.COLOR_BGR2GRAY)

        return letter_mask, shapeColor, letterColor

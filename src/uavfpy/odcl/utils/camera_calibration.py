import cv2
import os
import numpy as np
import argparse

'''
    Notes:
        -run this file while inside the calibration folder.
        -this assumes checkerboards folder exists with checkerboards images.
'''

def chessboardProcess(checkerPath: str, outfile = None, show_check=False, dim=(7,9)):
    '''

    '''
    checkerboard_list = os.listdir(checkerPath)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #dim
    m,n = dim


    #create points for calibration
    objp = np.zeros((m*n,3), np.float32)
    objp[:,:2] = np.mgrid[0:m,0:n].T.reshape(-1,2)

    objpoints = []
    imgpoints = []

    for path in checkerboard_list:
        if not path.endswith('.jpg'): continue

        #read img and convert to gray
        img = cv2.imread(checkerPath+path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Find chess board corners.
        ret, corners = cv2.findChessboardCorners(gray, (m,n), None)

        if ret==True: 

            #update points for calibration
            objpoints.append(objp)
            imgpoints.append(corners)

            #corners for drawing
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            
            #showing the cool checkerboard image w/ corners.
            if show_check:
                cv2.drawChessboardCorners(img, (m,n), corners2, ret)
                cv2.imshow('checkerboard', img)
                cv2.waitKey(0)
    cv2.destroyAllWindows()

    #these are all of the stuff generated for camera model
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if outfile != None:
        np.save(outfile, (ret,mtx,dist,rvecs,tvecs))
    return ret,mtx,dist,rvecs,tvecs



if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--checkerboard_path', required=True, help='path of checkerboards')
    arg.add_argument('--params_out', required=True, help='new file to save camera parameters')
    arg.add_argument('--show', required=True, help='show checkerboard visuals')
    opt = arg.parse_args()

    test = chessboardProcess(opt.checkerboard_path, outfile=opt.params_out, show_check=eval(opt.show))
    print('Finished Calibrating')
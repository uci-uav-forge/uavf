'''
GeoLocation Library
--------------------

Should include stufff like:
    - trig calculations
    - lidar or any form of depth estimation needec
    - any additional calculations to estimate relative distance

Requires:
    - pixel -> relative distance REQUIRES -> camera to be calibrated in vsutils.py
    - rotating x,y translations to absolute REQUIRES -> heading from UAV
    - getting depth REQUIRES -> altitude from UAV
    - doing any rotation w/ altitude REQUIRES -> rotation angles of gimbal

'''
import numpy as np
import cv2


EARTH_RADIUS = 6378137 #meters of radius of earth

def depth_givenAltitude(altitude, pitch=0,roll=0,yaw=0):
    '''
        Input:
        -------
            altitude: float (altitude of the UAV)
            pitch,roll,yaw: float (radians)
                    (represents 3 possible axis of rotations of gimbal)


        Output:
        --------
        depth = float (distance from the camera (depth)) in (m) meters.
    '''
    if pitch==roll==yaw==0:
        return altitude
    else:

        '''
            do math by hand if you want to reduce matrix multiplications to just one matrix.
        '''
        roll_mtx = np.array([ [np.cos(roll) , 0, -np.sin(roll)],
                              [ 0, 1, 0],
                              [np.sin(roll), 0, np.cos(roll)]])
        pitch_mtx = np.array([  [1, 0, 0],
                                [0, np.cos(pitch), -np.sin(pitch)],
                                [0, np.sin(pitch), np.cos(pitch)]])

        yaw_mtx = np.array([ [np.cos(yaw), np.sin(yaw), 0],
                             [-np.sin(yaw), np.cos(yaw), 0],
                             [0, 0, 1]])

        #determine final rotation matrix
        rot_mtx = yaw_mtx @ roll_mtx @ pitch_mtx #matrix multiplication
        
        alt_vector = np.array([[0],[0],[altitude]]) 
        rot_vec = np.matmul( rot_mtx, alt_vector )

        #get angle between 2 vectors
        angle = np.arccos( np.vdot(alt_vector.T, rot_vec) / np.linalg.norm(alt_vector) / np.linalg.norm(rot_vec ))
        #assuming ground is flat, we can use right triangle to get depth
        return altitude / np.cos(angle)


def get_relDist(pixel_coord, img_shape, focal, depth):
    '''
    Inputs:
    --------
        pixel_coord: (x,y) of the pixel
        img_shape: (h,w,3) of the shape
        focal: (x,y) of respective fo
    Outputs:
    --------
        returns x,y relative to camera
    '''
    px, py = pixel_coord
    h, w , _ = img_shape
    focalx, focaly = focal

    cx = w/2
    cy = h/2
    x = (px - cx) * depth / focalx
    y = (py - cy) * depth / focaly
    
    return x,y

def get_absCoord(tvecs, heading):
    '''
        Inputs:
        --------
            tvecs: (x,y) translation
            heading: yaw of the UAV
        Outputs:
        --------
            x: newly rotated x vector
            y: newly rotated y vector
    '''
    rad = heading * np.pi / 180
    rot = np.array([[np.cos(-rad), -np.sin(-rad)], \
                    [np.sin(-rad), np.cos(-rad) ]])
    x,y = tvecs

    return rot@x, rot@y 

def meters_to_gps(dx,dy):
    '''
        Input:
            dx: distance in x (in meters)
            dy: distance in y (in meters)
        Output:
            latitude: degrees
            longitude: degrees

        converts distances from meters to gps coordinates.
    '''
    y = (180/np.pi)*(dy/EARTH_RADIUS)
    x = (180/np.pi)*(dx/EARTH_RADIUS)/np.cos(np.pi/180 * dy)    

    #returns lat, lon
    return (y, x)

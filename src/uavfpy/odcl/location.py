'''
GeoLocation Library
--------------------

Should include:
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

EARTH_RADIUS = 6378137 #meters of radius of earth


class Geolocation:
    def __init__(self) -> None:
        self.focal_matrix = np.load("camera_intrinsics.npy") # Created after performing
        self.focal = (self.focal_matrix[0, 0], self.focal_matrix[1, 1])
        self.img_shape = (3000, 4000, 3)


    def get_uavPerspective(self, x,y, altitude, yaw=0, pitch=0, roll=0):
        """
        Inputs:
            x, y, z = from UAV in meters.
            altituded = from UAV in meters.
            yaw, pitch, roll = from UAV
        Output:

        """
        yaw = yaw * np.pi / 180;
        pitch = pitch * np.pi / 180;
        roll = roll * np.pi / 180;
        
        yaw_Matrix = np.array(
                        [[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0 , 1]
                        ],dtype=float);
        pitch_Matrix = np.array(
                        [[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]
                        ],dtype=float);
        roll_Matrix = np.array(
                        [[1,0,0],
                        [0,np.cos(roll), -np.sin(roll)],
                        [0,np.sin(roll), np.cos(roll)]
                        ],dtype=float);    
        
        # pitch -> roll -> yaw
        target_dist_Matrix = np.array([[x,y,altitude]]).T
        rotation_Matrix = pitch_Matrix @ roll_Matrix @ yaw_Matrix
        rotated_target_dist = rotation_Matrix @ target_dist_Matrix
        z_coord = target_dist_Matrix[2,:]
        projected = altitude / (z_coord) * rotated_target_dist # Column vector 1x3
        
        return (projected[0,:], projected[1,:], projected[2,:])


    def get_relDist(self, pixel_coord, img_shape, focal, depth):
        '''
        Inputs:
        --------
            pixel_coord: (x,y) of the pixel -- center of box relative to drawing on ground
            img_shape: (h,w,3) of the shape
            focal: (x,y) of respective fo -- calculted with camera calibration
            depth: float (distance from the camera (depth)) in (m) meters
        Outputs:
        --------
            returns x,y of bounding box center relative to camera on drone in meters
        '''
        px, py = pixel_coord
        h, w , _ = img_shape
        focalx, focaly = focal

        cx = w/2
        cy = h/2

        print("cx", cx, "cy", cy)

        x = (px - cx) * depth / focalx
        y = (py - cy) * depth / focaly
        
        return x,y


    def meters_to_gps(self, dx,dy):
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

    
    def compute(self, altitude, pitch, roll, yaw, gps_coord, pixel_coord):
        """
        Inputs:
            altitude and yaw: from Mavros data
            gps_coord: (lat, lon)
            pixel_coord: (x,y) of the pixel -- center of box relative to drawing on ground

        Output:
            New GPS coordinate in format (lat, lon) type tuple
        """

        comp_dx, comp_dy = self.get_relDist(pixel_coord, self.img_shape, self.focal, altitude)
        x, y, z = self.get_uavPerspective(comp_dx,comp_dy, altitude, yaw, pitch, roll)

        lat, lon = self.meters_to_gps(x, y)
        new_lat = gps_coord[0] + lat
        new_lon = gps_coord[1] + lon
        return (new_lat, new_lon)

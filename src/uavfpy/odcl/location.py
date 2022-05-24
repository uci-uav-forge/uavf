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
        self.focal_matrix = np.load("camera_intrinsic.npy") # Created after performing
        self.focal = (self.focal_matrix[0, 0], self.focal_matrix[1, 1])
        self.img_shape = (3000, 4000, 3)


    def quaternionToRotation(self, quat=None):
        q0,q1,q2,q3 = quat # quat = (w,x,y,z)
        if quat is None:
            return np.eye(3,dtype=float)

        Rot = np.array([[2*(q0**2+q1**2)-1, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
                        [2*(q1*q2+q0*q3), 2*(q0**2+q2**2)-1, 2*(q2*q3-q0*q1)],
                        [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 2*(q0**2+q3**2)-1]])
        
        return Rot
    def get_uavPerspective(self, x,y, altitude, quat=None):
        """
        Inputs:
            x, y, z = from UAV in meters.
            altitude = from UAV in meters.
            yaw, pitch, roll = from UAV
            quat = (x,y,z,w)
        Output:

        """
        
        # pitch -> roll -> yaw
        target_dist_Matrix = np.array([[x,y,altitude]]).T
        rotation_Matrix = self.quaternionToRotation(quat)
        rotated_target_dist = rotation_Matrix @ target_dist_Matrix
        z_coord = target_dist_Matrix[2,:][0]

        # TESTING
        print("Z-coordinate:", z_coord)

        if z_coord == 0:
            return (rotated_target_dist[0,:], rotated_target_dist[1,:], rotated_target_dist[2,:])

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

    
    def compute(self, altitude, quat, gps_coord, pixel_coord):
        """
        Inputs:
            altitude and yaw: from Mavros data
            gps_coord: (lat, lon)
            pixel_coord: (x,y) of the pixel -- center of box relative to drawing on ground

        Output:
            New GPS coordinate in format (lat, lon) type tuple
        """

        comp_dx, comp_dy = self.get_relDist(pixel_coord, self.img_shape, self.focal, altitude)
        x, y, z = self.get_uavPerspective(comp_dx,comp_dy, altitude, quat)

        lat, lon = self.meters_to_gps(x, y)
        new_lat = gps_coord[0] + lat
        new_lon = gps_coord[1] + lon
        return (new_lat, new_lon)

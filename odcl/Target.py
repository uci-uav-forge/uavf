

'''
Idea:
	After Object Detection Model extracts all of the targets detected,
	we will have to store it in a data structure that will contain all of the targets, and more data.

	2 classes will help accomplish this task in the pipeline.
		-TargetStore class that will store all of the targets and do any manipulation required of it.
		-Target class that will store Targets (not necessarily a copy of cropped image) and all the metadata that comes with it
			(GPS Location, Letter, Shape, ...etc.)
'''

class Target:
    def __init__(self, shape:str=None, letter:str=None, heading:str=None, shape_color:str=None, letter_color:str=None, gps:(float,float)=None):
        '''
            class attributes:
            -----------------
            shape -> specified shape of the target (string)
            letter -> specified letter of the target (char/string)
            heading -> specified cardinal orientation of the target (string) (all 8 directions, not just N,S,E,W)
            shape_color -> specified shape/background color of the target (string)
            letter_color -> specified letter color of the target (string)
            gps -> specified latitude,longitude of target. 2-tuple floats (latitude,longitude)
        '''
        self.shape = shape
        self.letter = letter
        self.heading = heading
        self.shape_color = shape_color
        self.letter_color = letter_color
        self.gps = gps


    #getter methods
    def getGPS(self):
        return self.gps
    def getShape(self):
        return self.shape
    def getLetter(self):
        return self.letter
    def getHeading(self):
        return self.heading
    def getShapeColor(self):
        return self.shape_color
    def getLetterColor(self):
        return self.letter_color
    #setter methods
    def setGPS(self,val):
        self.gps = val
    def setShape(self,val):
        self.shape = val
    def setLetter(self,val):
        self.letter = val
    def setHeading(self,val):
        self.heading = val
    def setShapeColor(self,val):
        self.shape_color = val
    def setLetterColor(self,val):
        self.letter_color = val


class TargetList:
    def __init__(self):
        '''
            the idea of this class is to store a list of targets.
            class attributes:
            -----------------
            target_list: just a list of targets.
        '''
        self.target_list = []
    def add_target(target: Target, arc_tolerance: float = 1e-2):
        '''
            input: target (from Target class in Target.py)
            output: nothing
                    self.target_list gets added

            *NOTE: as of right now, the arc tolerance I set is arbitrary,
            and should be modified when an appropriate one is found after
            flight test.
        '''
        if target.getGPS() == None:
            return None
        
        for item in target_list:
            (lat0,lon0) = item.getGPS()
            (lat1,lon1) = target.getGPS()
            '''
                arc calculation between 2 gps coords
                using the haversine formula
                will return the great-circle distance in meters.
            '''
            lat0 *= np.pi / 180
            lat1 *= np.pi / 180
            lon0 *= np.pi / 180
            lon1 *= np.pi / 180
            dlat = lat1 - lat0
            dlon = lon1 - lon0
            
            calc_a = np.sin(dlat/2)**2 + np.cos(lat0) + np.cos(lat1) * np.sin(dlon/2)**2
            calc_c = 2 * np.arctan2(np.sqrt(calc_a), np.sqrt(1-calc_a))
            arc = EARTH_RADIUS * c

            if np.abs(arc) < arc_tolerance:
                return None # don't add the target cuz there's already one w/ similar gps coordinates
        
        #if it is not close to none of the targets. (it is not a duplicate)
        #then append to list.
        self.target_list.append(target)

    def write_to_json(self):
        '''
            Objective of function:
                Write the target lists into a readable json
                submittable to competition.
        '''
        pass

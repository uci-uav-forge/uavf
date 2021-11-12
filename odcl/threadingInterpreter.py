import threading
from threading import Thread

'''
    Required Modules:
        interpreter.py
        tile_image.py
'''


'''
    Worker class for creating worker threads solely on processing tiled images.
    Written 11/3/2021. 
'''
class threadingInterpreter(Thread):
    def __init__(self, targetInterpreter, tiler, genLock, targetLock, worker_id):
        '''
            Attributes:
            -----------
            worker_id -> int
                        thread # of this particular worker object

            targetInterpreter -> targetInterpreter Object (in interpreter.py)
                                    contains a specific model instance to perform inference

            tiler -> Tiler Object (tile_image.py)
                    

            genLock, targetLock -> threading.lock object (threading module)
                        an object that can be used to Lock actions to prevent multiple threads
                        from accessing same variables. (common multithreading problem)


            img -> numpy array or opencv array [shape = (m,n,3)] mxn dimension
                    current image for thread to work on
            
            
            Inheritance:
            ------------
                Inherits from Thread which gives this multi-threading abilities.


            Notes:
            -------
             IMPORTANT: MAKE SURE THAT ALL WORKER THREAD INSTANCES SHARE THE SAME -> img, tileGen, and genLock.
             this is because the sole design of this multithreaded class is based around those assumptions and designed as such.

        '''
        self.id = worker_id
        #a targetInterpreter for the thread to use
        self.targetInterpreter = targetInterpreter
        self.tiler = tiler
        
        super().__init__(daemon=True) 
        #setting daemon = True means these threads are killed upon death of main.

        self.working = False #indicates whether thread is currently working
        self.img = None #represents current image being worked on by thread (pass by reference)
        self.tileGen = None
        self.targetList = None


        self.genLock = genLock
        self.targetLock = targetLock

    def start_work(self, img, tileGen, targetList): #this sets up the base image to tile on
        '''
        Parameters:
        ----------
        img: numpy array of np.uint8
                shape (m,n,3) where mxn is dimension of image.
        tileGen: generator of tile 

        Returns:
        --------
        None

        Description/Notes:
            the img is a pass-by-reference. therefore,
            no expensive calls are being made by calling this function.

            this image will be set as current image that worker thread will work on.
        '''
        self.img = img
        self.tileGen = tileGen
        self.targetList = targetList
        self.working = True


    def run(self):
        '''
        No Parameters
        No Returns

        Job:
        ------
        This thread's sole purpose is to wait for an image and generator from Tiler Object to be fed
        into this object.

        Afterwards, it will access the generator and perform inference on whatever tile it gets.
        for multiple threads, there is a Lock placed to prevent multiple access to generator at once.

        '''
        while True:
            while not self.working:
                pass #if nothing is in queue, don't do anything
            
            #after above while, we can say that there is an image to work on, and we tiles to pull from
            try:
                self.genLock.acquire() #prevent multiple access of generator
                (ymin, ymax) , (xmin, xmax), (i,j) = next(self.tileGen)
                self.genLock.release()

                self.targetInterpreter.interpret(self.img[ymin:ymax, xmin:xmax]) #this will update the targetInterpreter

                self.targetLock.acquire()
                for target in self.targetInterpreter.targets:
                    self.targetList.append(self.tiler.parse_localTarget(target, xmin, ymin))
                self.targetLock.release()
            except StopIteration: 
                #the only "error" this code will handle is StopIteration from the generator in Tile
                #this error signals that generator is empty.
                self.working = False
                if self.genLock.locked(): self.genLock.release()
                if self.targetLock.locked(): self.targetLock.release()




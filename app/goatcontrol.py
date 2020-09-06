
################
#### MOTOR ####

from adafruit_motorkit import MotorKit
import time
import numpy as np
#import picamera
#import apriltag
# from gps3.agps3threaded import AGPS3mechanism

class Car:
    """
    Car can do three different things:
    - Throttle = discrete {-1,0,1}: -1 is backward, 0 is standing and +1 is forward
    - Steer = continuous [-1,1] where -1 is going left and +1 is going right
    - cut: False/True
    """
    def __init__(self):
        self.kit = MotorKit()
        self.cutter = self.kit.motor4
        self.motor_left = [self.kit.motor1]
        self.motor_right = [self.kit.motor2]
        self.all_motors = self.motor_left + self.motor_right + [self.cutter]
        
    def stop(self, tid = 0):
        """
        The most important function. stops all motors!
        """
        time.sleep(tid)
        for m in self.all_motors:
            m.throttle = 0
            
    def _cut(self, action):
        if action == True:
            self.cutter.throttle = 1.0
        else:
            self.cutter.throttle = 0
                    
    def _motion(self, left=0, right=0):
        """
        Controls how much output on each set of motors.
        """        
        for m in self.motor_left:
            m.throttle = -left
        for m in self.motor_right:
            m.throttle = -right

    def drive(self, left = 0, right = 0, cut = False, *args, **kwargs):
        self._motion(left = left, right = right)
        self._cut(cut)

################
#### SENSORS ####

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# GET GPS SIGNAL
class GPSTracker:
    def __init__(self):
        self.agps_thread = AGPS3mechanism()  # Instantiate AGPS3 Mechanisms
        self.agps_thread.stream_data()  # From localhost (), or other hosts, by example, (host='gps.ddns.net')
        self.agps_thread.run_thread()  # Throttle time to sleep after an empty lookup, default '()' 0.2 two tenths of a second

    def __call__(self):
        return {
            
            'lat' : self.agps_thread.data_stream.lat,
            'lon' : self.agps_thread.data_stream.lon,
            'speed' : self.agps_thread.data_stream.speed,
        }

class GoatSensor:
    def __init__(self, apriltag=False):
        self.camera = picamera.PiCamera()
        self.resolution = (256,256)
        self.camera.resolution = self.resolution
        #self.camera.framerate = 24
        self.camera.crop = (0.0, 0.0, 1.0, 1.0)
        # Init buffer
        self.rgb = bytearray(self.camera.resolution[0] * self.camera.resolution[1] * 3)
        self.img = np.empty( self.resolution+ (3,), dtype=np.uint8)
        
        # APRILTAG INIT
        self.apriltag = apriltag
        if self.apriltag:
            self.detector_apriltag = apriltag.Detector()
        
        # GPS SENSOR
        self.gps = GPSTracker()
        
        time.sleep(2)
    def close(self):
        self.camera.close()
    def capture(self):
        self.camera.capture(self.img,use_video_port=True, format= 'rgb')
        return self.img

    def detect_apriltag(self):
        gray = rgb2gray(self.img).astype(np.uint8)
        apriltags = self.detector_apriltag.detect(gray)
        if len(apriltags) == 0:
            aptag = None
        elif len(apriltags)>0:
            obj = apriltags[0]
            aptag = {'corners' : obj.corners.astype("int"),
                    'center' : obj.center,
                    'centerpct' : obj.center / self.resolution
                    }
        return aptag
    
    def step(self, show_apriltag = True):
        state = {}
        state['image'] = self.capture()
        
        
        coord = self.gps()
        state.update(coord)
        if self.apriltag:
            state['apriltag'] = self.detect_apriltag()
            if show_apriltag & (state['apriltag'] is not None):
                corners = aptag['corners']
                state['image'][corners[0,1]:corners[2,1], corners[0,0]:corners[2,0],1] = 200
            
        return state

if __name__ == "__main__":
    print("Testing motor capabilities")
    car = Car()
    car._motion(-1,1)
    car.stop(tid=2)
    print("Done. exiting.")
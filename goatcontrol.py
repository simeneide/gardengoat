
################
#### MOTOR ####

from adafruit_motorkit import MotorKit
import time

class Car:
    """
    Car can do three different things:
    - Throttle = discrete {-1,0,1}: -1 is backward, 0 is standing and +1 is forward
    - Steer = continuous [-1,1] where -1 is going left and +1 is going right
    - cut: False/True
    """
    def __init__(self):
        self.kit = MotorKit()
        self.cutter = self.kit.motor3
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
            m.throttle = left
        for m in self.motor_right:
            m.throttle = right
    def _control_car(self, throttle = 0, steer = 0):
        if throttle == 1:
            left = throttle  #* max(steer,0)
            right = throttle #*max(-steer,0)
        if throttle == 0:
            left = steer
            right = -steer
        if throttle == -1:
            left = - 1
            right = -1
        self._motion(left = left, right = right)

    def drive(self, actiondict = None, throttle = 0, steer = 0, cut = False, duration = None):
        
        if actiondict is not None:
            throttle = actiondict['throttle']
            steer = actiondict['steer']
            cut = actiondict['cut']
        
        self._control_car(throttle = throttle, steer = steer)
        self._cut(cut)
        #self.stop(duration)

################
#### SENSORS ####

import time
import picamera
import numpy as np
import apriltag

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# GET GPS SIGNAL
from gps3.agps3threaded import AGPS3mechanism
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
    def __init__(self):
        self.camera = picamera.PiCamera()
        self.resolution = (256,256)
        self.camera.resolution = self.resolution
        #self.camera.framerate = 24
        self.camera.crop = (0.0, 0.0, 1.0, 1.0)
        # Init buffer
        self.rgb = bytearray(self.camera.resolution[0] * self.camera.resolution[1] * 3)
        self.img = np.empty( self.resolution+ (3,), dtype=np.uint8)
        
        # APRILTAG INIT
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
        state['apriltag'] = self.detect_apriltag()
        
        coord = self.gps()
        state.update(coord)
        
        if show_apriltag & (state['apriltag'] is not None):
            corners = aptag['corners']
            state['image'][corners[0,1]:corners[2,1], corners[0,0]:corners[2,0],1] = 200
            
        return state
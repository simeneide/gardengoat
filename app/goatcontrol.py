
################
#### MOTOR ####
import cv2, queue, threading, time
from adafruit_motorkit import MotorKit
import time
import numpy as np
import logging
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
import asyncio
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

class EmptySensor():
    def __init__(self, **kwargs):
        logging.info("inizalizing empty sensor")
    def step(self,*args, **kwargs):
        return {}
    def stop(self, *args, **kwargs):
        pass

class GoatSensor:
    def __init__(self, apriltag=False):
        self.loop = asyncio.new_event_loop()
        self.sensors = {}
        self.sensors['camera'] = GoatCamera()
        
        # APRILTAG INIT
        self.apriltag = apriltag
        if self.apriltag:
            self.detector_apriltag = apriltag.Detector()
        
        # GPS SENSOR
        #self.sensor['gps'] = GPSTracker()
        
        time.sleep(2)
        
    def __call__(self):
        result = self.loop.run_until_complete(self.fetch_async())
        return result

    async def fetch_async(self):
        state = {
            name : self.loop.run_in_executor(None, sensor)
            for name, sensor in self.sensors.items()}
        result = await gather_dict(state)
        return result

    def stop(self):
        logging.info("stopping sensors..")
        for key, val in self.sensors.items():
            try:
                val.stop()
                logging.info(f"stopped sensor: {key}.")
            except Exception as e:
                logging.info(f"FAILED to stop sensor: {key}. Exception:")
                print(e)
                
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

async def gather_dict(tasks: dict):
    async def mark(key, coro):
        return key, await coro

    return {
        key: result
        for key, result in await asyncio.gather(
            *(mark(key, coro) for key, coro in tasks.items())
        )
    }


class GoatCamera:
    """ 
    Module that constantly reads the camera and gives you the latest frame when called.
    grabbed from here: https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv
    """
    def __init__(self, name=0, width =480, height=640):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.run_thread=True
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while self.run_thread:
          ret, frame = self.cap.read()
          if not ret:
            break
          if not self.q.empty():
            try:
              self.q.get_nowait()   # discard previous (unprocessed) frame
            except queue.Empty:
              pass
          self.q.put(frame)

    def __call__(self):
        return self.q.get()
<<<<<<< HEAD
    
    def stop(self):
=======
    def stop(self):
        self.run_thread = False
>>>>>>> 8de3cc20962cf7bc7400c6cda59ce8d88d766df1
        self.cap.release()
if __name__ == "__main__":
    print("Testing motor capabilities")
    #car = Car()
    #car._motion(-1,1)
    #car.stop(tid=2)
    print("Motor test done.")

    print("Test sensors")
    sensor = GoatSensor()
    for i in range(5):
        print(sensor())
        time.sleep(1.0)
    print("sensor test done.")

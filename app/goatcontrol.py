
################
#### MOTOR ####
import cv2, queue, threading, time
import time
import numpy as np
import logging
import pyrealsense2 as rs
#import picamera
#import apriltag
# from gps3.agps3threaded import AGPS3mechanism
        
def output_list(pins, value):
    for pin in pins:
        GPIO.output(pin, value)

class Car:
    def __init__(self):
        self.gpio_map = {
            'left' : {
                'forward' : [6,26],
                'backward' : [13,20]
            },
            'right' : {
                'forward' : [5,19],
                'backward' : [12,16]
            }
        }
        GPIO.cleanup()
        GPIO.setmode(GPIO.BCM)
        for side, motors in self.gpio_map.items():
            for key, val in motors.items():
                print(f"{side}-{key}-{val}")
                GPIO.setup(val, GPIO.OUT)
        logging.info(f"driving controls initialized.")
        
    def drive(self, left=0, right=0, cut=False , *args, **kwargs):
        if left>=0:
            output_list(self.gpio_map['left']['forward'], left)
            output_list(self.gpio_map['left']['backward'], 0)
        if left<=0:
            output_list(self.gpio_map['left']['backward'], -left)
            output_list(self.gpio_map['left']['forward'], 0)
        if right>=0:
            output_list(self.gpio_map['right']['forward'], right)
            output_list(self.gpio_map['right']['backward'], 0)
        if right<=0:
            output_list(self.gpio_map['right']['backward'], -right)
            output_list(self.gpio_map['right']['forward'], 0)
            
    def close(self):
        GPIO.cleanup()
        
        
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
    def close(self, *args, **kwargs):
        pass

class GoatSensor:
    def __init__(self, apriltag=False):
        self.loop = asyncio.new_event_loop()
        self.sensors = {}
        self.sensors['realsense'] = DepthCamera()
        
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
        state = {name:
            self.loop.run_in_executor(None, sensor)
            for name, sensor in self.sensors.items()}
        state = await gather_dict(state)
        
        result = {}
        for topkey, d in state.items():
            for key, val in d.items():
                result[key] = val
        return result

    def close(self):
        logging.info("closing sensors..")
        for key, val in self.sensors.items():
            try:
                val.close()
                logging.info(f"closed sensor: {key}.")
            except Exception as e:
                logging.info(f"FAILED to close sensor: {key}. Exception:")
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

def coord2array(coord):
    return np.array([coord.x, coord.y, coord.z])

class DepthCamera:
    def __init__(self):
        self.pipe = rs.pipeline()
        self.fps=10
        self.resolution = {'x':640,'y':480}
        conf = rs.config()
        conf.enable_stream(rs.stream.accel)
        conf.enable_stream(rs.stream.gyro)
        #conf.enable_stream(rs.stream.depth)
        #conf.enable_stream(rs.stream.color)

        #Setup streaming and recording
        conf.enable_stream(rs.stream.depth, self.resolution['x'], 
                                    self.resolution['y'])
        conf.enable_stream(rs.stream.color, self.resolution['x'], 
                                    self.resolution['y'])
        self.profile = self.pipe.start(conf)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        for x in range(5):
          self.pipe.wait_for_frames()
        logging.info("depth camera initialized.")
        
    def __call__(self):
        frame = self.pipe.wait_for_frames()
        out = {}
        #out['frame'] = frame
        out['camera'] = np.asanyarray(frame.get_color_frame().get_data())
        out['depth_frame'] = np.asanyarray(frame.get_depth_frame().get_data())#*self.depth_scale
        out['pose'] = coord2array(frame[2].as_motion_frame().get_motion_data()) # unsure whether this is motion of accel
        out['acceleration'] = coord2array(frame[3].as_motion_frame().get_motion_data()) # unsure whether this is motion of accel
        return out
    def close(self):
        self.pipe.stop()

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

    def close(self):
        self.run_thread = False
        self.cap.release()
        
if __name__ == "__main__":
    print("Testing motor capabilities")
    car = Car()
    car.drive(-1,1, cut=False)
    time.sleep(1)
    car.drive(0,0, cut=True)
    time.sleep(1)
    car.close()
    print("Motor test done.")

    print("Test sensors")
    sensor = GoatSensor()
    for i in range(5):
        print(sensor())
        time.sleep(1.0)
    print("sensor test done.")

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
        self.motor_right = [self.kit.motor4]
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
        Revert sign because of how motors are attached.
        """        
        for m in self.motor_left:
            m.throttle = -left
        for m in self.motor_right:
            m.throttle = -right
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
        self._cut(actiondict['cut'])
        #self.stop(duration)
import datetime
import time
import pickle
import os
class SaveTransitions:
    def __init__(self, data_dir = "data"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.image_prev = None

    def save_step(self, action, newimage):
        
        event = {
            'timestamp' : datetime.datetime.now(),
            'action' : action,
            'image' : newimage,
            'image_prev' : self.image_prev,
        }

        filename = f'{self.data_dir}/{str(event["timestamp"]).replace(" ","_")}.pickle'
        with open(filename, "wb+") as handler:
            pickle.dump(event, handler)

        self.image_prev = newimage # update last image for next iteration


class Discretize_loop:
    def __init__(self, step_time= 0.2):
        self.step_time = step_time
        
    def start(self):
        self.start_time = datetime.datetime.now()
    
    def end(self):
        self.end_time = datetime.datetime.now()
        elapsed_time = (self.end_time - self.start_time).microseconds/1e6
        wait = self.step_time - elapsed_time
        
        if wait < 0:
            print(f"WARNING: Loop elapsed time is above step time: {elapsed_time} vs {self.step_time}")
        else:
            print(elapsed_time)
            time.sleep(wait)
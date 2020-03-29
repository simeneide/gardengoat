import datetime
import time
import pickle
import os
import logging
class SaveTransitions:
    def __init__(self, data_dir = "data"):
        self.data_dir = data_dir
        self.tub = f"tub_{datetime.datetime.now()}"
        self.save_dir = f"{self.data_dir}/{self.tub}"
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.image_prev_path = None

    def save_step(self, action, state):
        newimage = state['image']
        ts = datetime.datetime.now()
        filename_dat = f'{self.save_dir}/event_{str(ts).replace(" ","_")}.pickle'
        filename_next_img = f'{self.save_dir}/img_{str(ts).replace(" ","_")}.pickle'
        
        event = {
            'timestamp' : ts,
            'action' : action,
            'image_next_path' : filename_next_img,
            'image_path' : self.image_prev_path,
        }
        # Add sensors to event (execpt image):
        for key, val in state.items():
            if key != "image":
                event[key] = val

        with open(filename_dat, "wb+") as handler:
            pickle.dump(event, handler)
            
        with open(filename_next_img, "wb+") as handler:
            pickle.dump(newimage, handler)
            
        self.image_prev_path = filename_next_img # update last image for next iteration


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
            logging.info(f"WARNING: Loop elapsed time is above step time: {elapsed_time} vs {self.step_time}")
        else:
            print(elapsed_time)
            time.sleep(wait)
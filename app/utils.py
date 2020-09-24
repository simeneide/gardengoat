import datetime
import time
import pickle
import os
import logging
import cv2
class SaveTransitions:
    def __init__(self, data_dir = "data"):
        self.data_dir = data_dir
        self.tub = f"tub_{datetime.datetime.now().strftime('%Y-%m-%d.%H:%M:%S')}"
        self.save_dir = f"{self.data_dir}/{self.tub}"
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save_step(self, action, state):
        if state.get('camera') is None:
            logging.warning("Empty image state when saving")
        if state.get("step") is None:
            logging.warning("Step number is not set")
            
        ts = datetime.datetime.now()
        filename_dat = f'{self.save_dir}/event_{state.get("step")}.pickle'
        filename_img = f'{self.save_dir}/img_{state.get("step")}.jpeg'
        
        
        event = {
            'timestamp' : ts,
            'action' : action,
            'image_path' : filename_img,
        }
        
        # Add sensors to event (execpt image):
        for key, val in state.items():
            if key != "camera":
                event[key] = val
        
        # SAVE
        with open(filename_dat, "wb+") as handler:
            pickle.dump(event, handler)
            
        cv2.imwrite(filename=filename_img, img=state['camera'])

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
            time.sleep(wait)
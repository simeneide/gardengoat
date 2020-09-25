import datetime
import time
import pickle
import os
import logging
#import cv2
import numpy as np
def loader(path):
    with open(path, "rb") as handle:
        dat = pickle.load(handle)
    return dat

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
        filename_event = f'{self.save_dir}/event_{state.get("step")}.pickle'
        filename_data = f"data_{state.get('step')}.pickle"
        path_data = f'{self.save_dir}/{filename_data}'
        
        event = {
            'timestamp' : ts,
            'action' : action,
            'data_path' : filename_data,
        }
        
        # Add some sensors to event (execpt image):
        for key in ['step']:
            if  state.get(key):
                event[key] = state.get(key)
        
        # SAVE
        with open(filename_event, "wb+") as handler:
            pickle.dump(event, handler)
            
        with open(filename_data, "wb+") as handler:
            pickle.dump(state, handler)

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
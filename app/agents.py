import webserver
import models
import random
import datetime
import logging
import time

class GoatAgent:
    def __init__(self):
        self.idx2action = {
            0 : 'stop',
            1 : 'backandturn',
            2 : 'forward',
            3 : 'left',
            4 : 'backward'
        }

        self.agent_ui = webserver.Webagent()
        self.agent_ai = models.ConvNet()


        
        self.mode = "human"
        self.active_option = None
        
    def step(self, *args, **kwargs):
        """
        # priority:
        1. query ui
        2. conduct backandturn
        3. query ai
        """
        action = self.agent_ui(*args, **kwargs)

        
        # Execute UI straight away if actions are given:
        if (action.get("action") not in ['AI','backandturn']):
            self.mode = "human" #cancel any other mode
            self.active_option = None
            self.starttime_backandturn = False
            action['mode'] = self.mode
            if self.active_option:
                action['active_option'] = self.active_option.name
            return action
        elif action.get("action") == "AI":
            self.mode = "AI"
        
        if self.active_option:
            action, done = self.active_option()
            print(done)
            if done:
                self.active_option=None
                if self.mode == "human":
                    self.agent_ui.key=None
                
        elif self.mode =="AI":
            action = self.agent_ai.step(*args, **kwargs)

        if (action.get("action") == "backandturn") & (not self.active_option):
            self.active_option = BackAndTurn()
            
        
        action['mode'] = self.mode
        if self.active_option:
            action['active_option'] = self.active_option.name
        return action

class BackAndTurn:
    def __init__(self, back_time=None, rot_time=None):
        # Backandturn parameters:
        self.name = "BackAndTurn"
        self.starttime_backandturn = datetime.datetime.now()
        self.back_time = None
        self.rot_time = None
        
        if back_time is None:
            self.back_time = 2
        if rot_time is None:
            self.rot_time = random.randint(2,4)
        logging.info(f"Starting backandturn manouvre..: back={self.back_time}, turn={self.rot_time}")

    def __call__(self):
        """ A special move that override other moves once started. """
        time_since_started = (datetime.datetime.now()-self.starttime_backandturn).seconds
        if time_since_started < self.back_time:
            return {'left' : -1, 'right' : -1, 'action' : "backandturn"}, False
        elif time_since_started < (self.back_time + self.rot_time):
            return {'left' : -1, 'right' : 1, 'action' : "backandturn"}, False
        else:
            logging.info("Done with backandturn move.")
            return {'left' : 0, 'right' : 0}, True
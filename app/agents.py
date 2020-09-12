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
        self.agent_ai = models.GreenNet()

        # Backandturn parameters:
        self.starttime_backandturn = False
        self.back_time = None
        self.rot_time = None

    def start_backandturn(self, back_time=None, rot_time=None):
        
        self.starttime_backandturn = datetime.datetime.now()
        if back_time is None:
            self.back_time = 2 #random.randint(1,3)
        if rot_time is None:
            self.rot_time = random.randint(2,4)
        logging.info(f"Starting backandturn manouvre..: back={self.back_time}, turn={self.rot_time}")
        return self.action_backandturn()
    
    def action_backandturn(self):
        """ A special move that override other moves once started. """
        time_since_started = (datetime.datetime.now()-self.starttime_backandturn).seconds
        if time_since_started < self.back_time:
            return {'left' : -1, 'right' : -1}
        elif time_since_started < (self.back_time + self.rot_time):
            return {'left' : -1, 'right' : 1}
        else:
            logging.info("Done with backandturn move.")
            self.starttime_backandturn=False
            self.agent_ui.key=None ##### NBNBNB THIS THING IS CANCELLING AI MODE!
            return {'left' : 0, 'right' : 0}
        
    def step(self, *args, **kwargs):
        """
        # priority:
        1. query ui
        2. conduct backandturn
        3. query ai
        """
        action = self.agent_ui(*args, **kwargs)
        
        # Execute UI if left or right is given:
        if (action.get("left", False) | action.get("right", False) | action.get("cut", False)):
            return action
        
        # If we are in backturn move do that:
        if self.starttime_backandturn:
            return self.action_backandturn()
        elif action.get("action") == "AI":
            action = self.agent_ai.step(*args, **kwargs)
            
        if (action.get("action") == "backandturn") & (self.starttime_backandturn is False):
            return self.start_backandturn()
        else:
            return action

        
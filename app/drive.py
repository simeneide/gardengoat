import io
import time
import goatcontrol
import random
import utils
import numpy as np
import logging
#import agents
import webserver
logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')

car = goatcontrol.Car()
recorder = utils.SaveTransitions()
discrete_timer = utils.Discretize_loop(0.2)

# Init goatsensor
goatsensor = goatcontrol.GoatSensor()

# Main loop
exitFlag = True

import models
#% SET DRIVING AGENT
import agents
agent = agents.GoatAgent()
step = 0
logging.info("Starting driving..")
#### ---–--------
#### DRIVING LOOP
try:
    while(exitFlag):
        discrete_timer.start()
        ## OBSERVE
        state = goatsensor()
        state['step'] = step
        
        ## CHOOSE ACTION
        action = agent.step(state)
        logging.info(action)
        
        if action.get("shutdown"):
            exitFlag=False
            
        ## EXECUTE ACTION
        car.drive(**action)

        ### RECORD EVENTS ###
        if action != {}: # i.e any action was taken
            recorder.save_step(
                action = action, 
                state = state)

        step += 1
        discrete_timer.end()
except (KeyboardInterrupt, SystemExit):
    logging.info("Shutting down")
    pass
#except Exception as e:
#    print("something wrong:")
#    print(e)

# shutdown commands:
car.stop()
goatsensor.stop()
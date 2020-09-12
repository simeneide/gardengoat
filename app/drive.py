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

#% SET DRIVING AGENT
import agents
agent = agents.GoatAgent()
logging.info("Starting driving..")
#### ---–--------
#### DRIVING LOOP
state = None # Init state
try:
    while(exitFlag):
        discrete_timer.start()

        action = agent.step(state)
        logging.info(action)
        if action.get("shutdown"):
            exitFlag=False
        car.drive(**action)

        ## CONTROL SEQUENCE
        state = goatsensor()

        ### RECORD EVENTS ###
        if action != {}: # i.e any action was taken
            recorder.save_step(
                action = action, 
                state = state)

        discrete_timer.end()
except (KeyboardInterrupt, SystemExit):
    print("Shutting down")
    pass
#except Exception as e:
#    print("something wrong:")
#    print(e)

car.stop()
goatsensor.stop()
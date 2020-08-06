import io
import time
# Init car
import goatcontrol
import random
import utils
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')

car = goatcontrol.Car()
recorder = utils.SaveTransitions()
discrete_timer = utils.Discretize_loop(0.2)

visualize=False

# Init goatsensor
goatsensor = goatcontrol.GoatSensor()

if visualize:
    from pygame.locals import *
    import pygame
    # Init pygame
    pygame.init()
    screen = pygame.display.set_mode(goatsensor.resolution)
    pygame.display.set_caption('GardenGoat')
    x = (screen.get_width() - goatsensor.resolution[0]) / 2
    y = (screen.get_height() - goatsensor.resolution[1]) / 2
else:
    pygame=None

# Main loop
exitFlag = True

#% SET DRIVING AGENT
import agents
import webserver
agent = webserver.Webagent()# agents.keyboard_local #agents.TorchAction()#keyboard_local

#### ---–--------
#### DRIVING LOOP
state = {} # Init state
try:
    while(exitFlag):
        discrete_timer.start()
        print("start")
        if visualize:
            for event in pygame.event.get():
                if(event.type is pygame.MOUSEBUTTONDOWN or 
                    event.type is pygame.QUIT):
                    exitFlag = False
            screen.fill(0)

        action = agent(state, pygame)
        if action.get("stop"):
            exitFlag=False
        car.drive(**action)

        ## CONTROL SEQUENCE
        state = goatsensor.step()
        img = state['image']

        if visualize & (random.random() > 0.5):
            img_pygame = pygame.surfarray.make_surface(np.swapaxes(img,0,1))
            if img_pygame:
                screen.blit(img_pygame, (x,y))
                pygame.display.update()

        ### RECORD EVENTS ###
        if sum([abs(val) for key, val in action.items()]) > 0: # i.e any action was taken
            recorder.save_step(
                action = action, 
                state = state)

        discrete_timer.end()
except (KeyboardInterrupt, SystemExit):
    print("Shutting down")
    pass

car.stop()
goatsensor.close()
if visualize:
    pygame.display.quit()

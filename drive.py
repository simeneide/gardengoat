import pygame
import io
import time
# Init car
import goatcontrol
from pygame.locals import *
import utils
import numpy as np
car = goatcontrol.Car()
recorder = utils.SaveTransitions()
discrete_timer = utils.Discretize_loop(0.2)

# Init goatsensor
goatsensor = goatcontrol.GoatSensor()
# Init pygame
pygame.init()
screen = pygame.display.set_mode(goatsensor.resolution)
pygame.display.set_caption('GardenGoat')
x = (screen.get_width() - goatsensor.resolution[0]) / 2
y = (screen.get_height() - goatsensor.resolution[1]) / 2

# Main loop
exitFlag = True

#% SET DRIVING AGENT
import agents
agent = agents.keyboard_local

# INIT GPS
import gpstracker
gps = gpstracker.GPSTracker()

#### ---–--------
#### DRIVING LOOP
state = {} # Init state
while(exitFlag):
    discrete_timer.start()
    
    for event in pygame.event.get():
        if(event.type is pygame.MOUSEBUTTONDOWN or 
            event.type is pygame.QUIT):
            exitFlag = False
    screen.fill(0)
    try:
        action = agent(state)
    except:
        print("Agent didnt work.. Stay still.")
        action = {
        'throttle' : 0,
        'steer' : 0,
        'cut' : 0
    }
    car.drive(actiondict=action)
    
    ## CONTROL SEQUENCE
    state = goatsensor.step()
    img = state['image']
    
    
    time.sleep(0.3)
    #car.stop()
    #time.sleep(0.1)
    
    import random
    if random.random() > 0.5:
        img_pygame = pygame.surfarray.make_surface(np.swapaxes(img,0,1))
        if img_pygame:
            screen.blit(img_pygame, (x,y))
            pygame.display.update()
        
    ### RECORD EVENTS ###
    if sum([abs(val) for key, val in action.items()]) > 0: # i.e any action was taken
        recorder.save_step(
            action = action, 
            newimage = img)
        
    discrete_timer.end()
    print(state['lat'], state['lon'])

goatsensor.close()
pygame.display.quit()
car.stop()
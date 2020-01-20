import pygame
import io

# Init car
import goatcontrol
from pygame.locals import *
import utils
import numpy as np
car = goatcontrol.Car()
recorder = utils.SaveTransitions()
discrete_timer = utils.Discretize_loop(0.2)

# Init camera
camera = goatcontrol.GoatCam()
# Init pygame 
pygame.init()
screen = pygame.display.set_mode(camera.resolution)
pygame.display.set_caption('GardenGoat')
x = (screen.get_width() - camera.resolution[0]) / 2
y = (screen.get_height() - camera.resolution[1]) / 2


# DEFAULT PARS (ie dont drive, steer or cut)
def keyboard_control():
    keys = pygame.key.get_pressed()
    action = {
        'throttle' : 0,
        'steer' : 0,
        'cut' : 0
    }

    if keys[K_w]:
        action['throttle'] = 1
    elif keys[K_s]:
        action['throttle'] = -1
    elif keys[K_a]:
        action['steer'] = -1
    elif keys[K_d]:
        action['steer'] = 1

    if keys[K_SPACE]:
        action['cut'] = True

    return action

import time
def drive_to_tag(state):
    action = {
        'throttle' : 0,
        'steer' : 0,
        'cut' : 0
    }
    apriltag = state['apriltag']
    if apriltag:
        angle = (apriltag['centerpct'][0]-0.5).round(3)
        if abs(angle) <0.03:
            action['throttle'] = 1
            action['steer'] = 0
        else:
            action['throttle']=0
            action['steer'] = np.sign(angle)
            
        print(f"throttle = {action['throttle']}, steer = {action['steer']}, angle: {angle}")
    else:
        action['throttle'] = 0.0
        action['steer'] = 1
        print(f"throttle = {action['throttle']}, steer = {action['steer']}, searching..")
    return action

# Main loop
exitFlag = True
while(exitFlag):
    discrete_timer.start()
    
    for event in pygame.event.get():
        if(event.type is pygame.MOUSEBUTTONDOWN or 
            event.type is pygame.QUIT):
            exitFlag = False
    screen.fill(0)
    
    
    ## CONTROL SEQUENCE
    state = camera.step()
    img = state['image']
    
    action = drive_to_tag(state)
    car.drive(actiondict=action)
    time.sleep(0.3)
    car.stop()
    time.sleep(0.05)
    
    img_pygame = pygame.surfarray.make_surface(np.swapaxes(img,0,1))

    #img = pygame.surfarray.array3d(img_pygame).swapaxes(0,1)
    
    if img_pygame:
        screen.blit(img_pygame, (x,y))
        pygame.display.update()

    

        
    ### RECORD EVENTS ###
    if sum([abs(val) for key, val in action.items()]) > 0: # i.e any action was taken
        recorder.save_step(
            action = action, 
            newimage = img)
        
    discrete_timer.end()

camera.close()
pygame.display.quit()
car.stop()
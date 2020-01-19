#import picamera
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
camera.res
# Init pygame 
pygame.init()
screen = pygame.display.set_mode(camera.res)
pygame.display.set_caption('GardenGoat')
x = (screen.get_width() - camera.res[0]) / 2
y = (screen.get_height() - camera.res[1]) / 2


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

# Main loop
exitFlag = True
while(exitFlag):
    discrete_timer.start()
    
    for event in pygame.event.get():
        if(event.type is pygame.MOUSEBUTTONDOWN or 
            event.type is pygame.QUIT):
            exitFlag = False
    screen.fill(0)
    
    vision_output = camera.step()
    img = vision_output['image']

    img_pygame = pygame.surfarray.make_surface(np.swapaxes(img,0,1))

    #img = pygame.surfarray.array3d(img_pygame).swapaxes(0,1)
    
    if img_pygame:
        screen.blit(img_pygame, (x,y))
        pygame.display.update()

    
    action = keyboard_control()
    car.drive(actiondict=action)
        
    ### RECORD EVENTS ###
    if sum([abs(val) for key, val in action.items()]) > 0: # i.e any action was taken
        recorder.save_step(
            action = action, 
            newimage = img)
        
    discrete_timer.end()

camera.close()
pygame.display.quit()
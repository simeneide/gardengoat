#### ---–--------
##### DIFFERENT DRIVING AGENTS ######
# DEFAULT PARS (ie dont drive, steer or cut)
import keyboard
def keyboard_local(*args, **kwargs):
    key = keyboard.read_key()
    print(key)
    action = {
        'left' : 0,
        'right' : 0,
        'cut' : 0
    }
    if key == "esc":
        action['stop'] = True
    elif key == "w":
        action['left'] = 1
        action['right'] = 1
    elif key == "s":
        action['left'] = -1
        action['right'] = -1
    elif key == "a":
        action['left'] = -1
        action['right'] = 1
    elif key == "d":
        action['left'] = 1
        action['right'] = -1
    elif key == "q":
        action['left'] = 0.5
        action['right'] = 1
    elif key == "e":
        action['left'] = 1
        action['right'] = 0.5
    if key == " ":
        action['cut'] = True
    return action

#from pygame.locals import *
def keyboard_control_pygame(state, pygame, *args, **kwargs):
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


## LOAD MODEL
import torch
import model_drive
class TorchAction:
    def __init__(self):

        self.model = torch.load("model.pickle")
        
    def __call__(self,state, *args, **kwargs):
        img = state.get("image")
        action = {
            'left' : 0,
            'right' : 0,
            'cut' : 0
        }

        if img is None:
            return action
        yhat = self.model(model_drive.tr(img).unsqueeze(0))
        action_cat = int(yhat.argmax())

        if action_cat==0:
            action['left'] = 1
            action['right'] = 1
        elif action_cat==1:
            action['left'] = -1
            action['right'] = -1
        elif action_cat==2:
            action['left'] = -1
            action['right'] = -0.5
        elif action_cat==3:
            action['left'] = -0.5
            action['right'] = -1
        return action

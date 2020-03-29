#### ---–--------
##### DIFFERENT DRIVING AGENTS ######
# DEFAULT PARS (ie dont drive, steer or cut)
import keyboard
def keyboard_local(*args, **kwargs):
    key = keyboard.read_key()
    print(key)
    action = {
        'throttle' : 0,
        'steer' : 0,
        'cut' : 0
    }

    if key == "w":
        action['throttle'] = 1
    elif key == "s":
        action['throttle'] = -1
    elif key == "a":
        action['steer'] = -1
    elif key == "d":
        action['steer'] = 1

    if key == " ":
        action['cut'] = True
    return action

def keyboard_control_pygame(*args, **kwargs):
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
#import torch
#import model_drive
#model = torch.load("model.pickle")

def action_model(state):
    img = state.get("image")
    action = {
        'throttle' : 0,
        'steer' : 0,
        'cut' : 0
    }
    
    if img is None:
        return action
    yhat = model(model_drive.tr(img).unsqueeze(0))
    action_cat = int(yhat.argmax())

    if action_cat==0:
        action['throttle'] = 1
    elif action_cat==1:
        action['throttle'] = -1
    elif action_cat==2:
        action['steer'] = -1
    elif action_cat==3:
        action['steer'] = 1
    return action

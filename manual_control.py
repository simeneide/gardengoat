import picamera
import pygame
import io

# Init pygame 
pygame.init()
screen = pygame.display.set_mode((256,256))
pygame.display.set_caption('GardenGoat')

# Init camera
camera = picamera.PiCamera()
camera.resolution = (256, 256)
camera.crop = (0.0, 0.0, 1.0, 1.0)

x = (screen.get_width() - camera.resolution[0]) / 2
y = (screen.get_height() - camera.resolution[1]) / 2

# Init buffer
rgb = bytearray(camera.resolution[0] * camera.resolution[1] * 3)


# Init car
import goatcontrol
car = goatcontrol.Car()
from pygame.locals import *

# Main loop
exitFlag = True
while(exitFlag):
    for event in pygame.event.get():
        if(event.type is pygame.MOUSEBUTTONDOWN or 
           event.type is pygame.QUIT):
            exitFlag = False

    stream = io.BytesIO()
    camera.capture(stream, use_video_port=True, format='rgb')
    stream.seek(0)
    stream.readinto(rgb)
    stream.close()
    img = pygame.image.frombuffer(rgb[0:
          (camera.resolution[0] * camera.resolution[1] * 3)],
           camera.resolution, 'RGB')

    screen.fill(0)
    if img:
        screen.blit(img, (x,y))

    pygame.display.update()
    
    keys = pygame.key.get_pressed()
    #print(keys[K_w])
    if keys[K_w]:
        car.drive(throttle=1)
    elif keys[K_s]:
        car.drive(throttle=-1)
    elif keys[K_a]:
        car.drive(steer=-1)
    elif keys[K_d]:
        car.drive(steer=1)
    else:
        car.stop()
    if keys[K_SPACE]:
        car._cut(True)
    else:
        car._cut(False)
        
camera.close()
pygame.display.quit()
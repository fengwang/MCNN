import cv2
import imageio
import mss
from time import sleep
import numpy as np

def wait(seconds=1):
    sleep(seconds)

#mon = {"top": 0, "left": 0, "width": 3840, "height": 2160}
mon = {"top": 0, "left": 0, "width": 1920, "height": 1080}
sct = mss.mss()
cap = cv2.VideoCapture(0)

dumped = False

def screen_get():
    global mon
    global sct
    img = np.asarray(sct.grab(mon))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.asarray( img, dtype='uint8' )

def camera_get():
    global cap
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    global dumped
    if dumped == False:
        dumped = True
        imageio.imsave( 'firstframe.png', frame )
    return np.asarray(frame, dtype='uint8')

def get_camera_and_screen(frames=16, interval=1.0):
    screens = []
    cameras = []
    for idx in range( frames ):
        screens.append( screen_get() )
        cameras.append( camera_get() )
        wait(interval)

    return ( np.asarray( screens, dtype='uint8' ), np.asarray( cameras, dtype='uint8' ) )

if __name__ == '__main__':
    sleep( 30 ) # 30s moving laptop to the laptop-standup facing the door
    for idx in range( 4 ):
        frames =  256
        interval = 0.5
        screens, cameras = get_camera_and_screen(frames, interval=interval)
        print( f'Video gen: all files generated.--{idx}-{frames}' )
        np.savez_compressed( f'./{frames}_{idx}screens_cameras.npz', screens=screens, cameras=cameras )



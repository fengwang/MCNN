# this script dump the camera and screenshot images from the stored np array
import numpy as np
import imageio

def gen(index):
    path = f'/run/media/feng/Samsung USB/door_mirror/{index}_256_screens_cameras.npz'
    data = np.load(path)
    sc = data['screens']
    ca = data['cameras']
    for idx in range( 256 ):
        imageio.imsave( f'./{index}/{str(idx).zfill(3)}_screenshot.png', sc[idx] )
        imageio.imsave( f'./{index}/{str(idx).zfill(3)}_cameracaptured.png', ca[idx] )


for index in range(4):
    gen(index)


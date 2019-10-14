import numpy as np
import imageio
import glob
import random
from simulate import cal_defocus
from multiprocessing import Pool, cpu_count

if __name__ == '__main__':
    wave_length = 0.525
    refractive_index = 1.518
    numerical_aperture = 1.35
    fsmooth = 2.0
    alpha = 0.05
    maximum_phase_range = 1.0
    scales = (0.156250, 0.156250)
    depths = [-500.       , -385.93384  , -297.88998  , -229.93166  , -177.4768   , -136.98866  , -105.73713  ,  -81.615074 , -62.99606  ,  -48.624622 ,  -37.531776 ,  -28.969576 , -22.36068  ,  -17.25949  ,  -13.322042 ,  -10.282856 , -7.9370055,   -6.1263185,   -4.7287083,   -3.6499374, -2.817269 ,   -2.1745594,   -1.6784723,   -1.2955587, -1.       ,    0.       ,    1.       ,    1.2955587, 1.6784723,    2.1745594,    2.817269 ,    3.6499374, 4.7287083,    6.1263185,    7.9370055,   10.282856 , 13.322042 ,   17.25949  ,   22.36068  ,   28.969576 , 37.531776 ,   48.624622 ,   62.99606  ,   81.615074 , 105.73713  ,  136.98866  ,  177.4768   ,  229.93166  , 297.88998  ,  385.93384  ,  500.       ]

    from random import shuffle
    images = glob.glob( '/raid/feng/pictures/*/*.png' )
    total = len(images)
    shuffle( images )
    output_dir = '/home/feng/cache/simulated_defocus_51_noisy_se/'

    def norm( img ):
        return (img-np.amin(img))/(np.amax(img)-np.amin(img)+1.0e-10)

    def gen_amplitude( img ):
        tmp = (random.random()-0.5)*0.4*norm(img) + 0.8
        return np.asarray( tmp/(np.amax(tmp)+1.0e-10), dtype='float32' )

    # central crop, [768x768] -> [512x512]
    def transform( img ):
        ans = img[128:896, 128:896]
        ans = np.sqrt( np.abs(ans) )
        ans = np.random.random( ans.shape ) * 0.05 + ans * 0.95
        return np.reshape( ans, ans.shape+(1,) ).astype('float32')

    def t_transform( img ):
        ans = img[128:896, 128:896]
        ans = norm(np.abs(ans))
        return np.reshape( ans, ans.shape+(1,) ).astype('float32')

    def make_simulation( idx ):
        phases = imageio.imread( images[idx] )
        phases = np.asarray( phases, dtype='float32' ) / ( np.amax( phases ) + 1.0e-10 )
        amplitude = imageio.imread( images[total-idx-1] )
        amplitude = np.asarray( amplitude, dtype='float32' ) / ( np.amax( phases ) + 1.0e-10 )
        amplitude = gen_amplitude( amplitude ) # or just pass phases here
        results = []
        for depth in depths:
            astigmatic_defocus = depth
            defocus = cal_defocus( phases, alpha, numerical_aperture, wave_length, refractive_index, astigmatic_defocus, scales, maximum_phase_range, [dim>>1 for dim in phases.shape], 'none', amplitude, fsmooth )
            results.append( transform(defocus) )
        record = np.concatenate( [t_transform(amplitude), t_transform(phases)]+results, axis=-1 )
        np.save( f'{output_dir}{idx}', record )

    max_to_simulate = min(1 * 1024 * 1024, len(images))
    indices = [ i for i in range( min( max_to_simulate, total ) ) ]
    with Pool(cpu_count()) as p:
        p.map( make_simulation, indices )


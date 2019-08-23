from glob import glob
import numpy as np
from random import uniform
from math import exp
import random
import sys
from random import shuffle
from skimage.measure import block_reduce
import cv2
import imageio

openimagenet_images = glob( '/run/media/feng/0d01a721-50e2-4d22-ac51-9d93d679474c/raid_backup/pictures/openimagenet.v5/*.jpg' )

def convolution( image, kernel ):
    assert 2 == len(image.shape)
    assert 2 == len(kernel.shape)
    return cv2.filter2D( src=image, kernel=kernel, ddepth=-1)

# imagenet images
def path( concreate_path = None ):
    ans = openimagenet_images
    shuffle( ans )
    return ans

def generator():
    image_paths = path()
    for current_path in image_paths:
        image = np.asarray( imageio.imread( current_path ), dtype='float32' )
        if np.sum( np.isinf(image) ) or np.sum( np.isnan(image) ):
            continue
        if len(image.shape) == 3:
            image = (image[:,:,0] + image[:,:,1] + image[:,:,2]) / 3.0
        print( f'loading image from {current_path}', end='\r' )
        yield image

def gray( images_to_generate=1, dimension=(512, 512) ):
    assert 2 == len(dimension), 'only 2D dimension supported'
    row, col = dimension
    images = np.zeros( (images_to_generate,) + dimension )
    images_loaded = 0
    for img in generator():
        r, c = img.shape
        if r >= row  and c >= col:
            scale_factor = min( int(r/row), int(c/col) )
            frow, fcol = scale_factor*row, scale_factor*col
            sr, sc = random.randint(0, r-frow), random.randint(0, c-fcol)
            selected_img = img[sr:sr+frow, sc:sc+fcol]
            images[images_loaded] = block_reduce(selected_img, block_size=(scale_factor, scale_factor), func=np.mean)
            images_loaded += 1
        if images_loaded >= images_to_generate:
            break

    return images

def calculate_kernel( sigma=10.0, dimension=(256,256) ):
    kernel = np.zeros( dimension )
    row, col = dimension
    for r in range( row ):
        for c in range( col ):
            offset_x = r - (row>>1)
            offset_y = c - (col>>1)
            kernel[r][c] = exp( - (offset_x*offset_x+offset_y*offset_y) / (sigma+sigma) )
    return kernel

def generate_random_coordinates( dimension=(512,512), columns=50, pixel_interval=32, max_intensity=5 ):
    intensities = np.random.random_integers( 1, max_intensity, columns )

    row, col = dimension
    r_, c_ = int(row/pixel_interval), int(col/pixel_interval)

    compressed_coords = np.random.random_integers( 0, r_*c_-1, r_*c_ )
    compressed_2d_coords = np.zeros( (r_, c_ ) )
    proceed = 0
    for idx in range( r_*c_ ):
        if proceed >= columns: # done
            break
        r, c = int( compressed_coords[idx]/r_), compressed_coords[idx]%r_
        if compressed_2d_coords[r][c] < 0.5:
            compressed_2d_coords[r][c] = intensities[proceed]
            proceed += 1

    random_offset = np.random.rand( columns, 2 ) * 0.25 - 0.125

    random_coords = np.zeros( (row, col) )
    proceed = 0
    for r in range(r_):
        for c in range(c_):
            if compressed_2d_coords[r][c] > 0.5:
                random_coords[int((r+random_offset[proceed][0])*pixel_interval)][int((c+random_offset[proceed][1])*pixel_interval)] = compressed_2d_coords[r][c]
                proceed += 1
    return random_coords

def remove_boundary( image, boundary=16 ):
    row, col = image.shape
    ans = np.zeros( (row, col) )
    ans[boundary:row-boundary, boundary:col-boundary] = image[boundary:row-boundary, boundary:col-boundary]
    return ans

def fake_image( dimension=(512, 512), range_of_columns=(20, 50), kernel=None, sigma=80.0, max_intensity = 5, offset=32, pixel_interval=32, save_path=None ):
    columns = int( random.uniform(*range_of_columns) )
    row, col = dimension
    row_coords = np.random.random_integers( offset, row-offset, columns )
    col_coords = np.random.random_integers( offset, col-offset, columns )
    intensities = np.random.random_integers( 1, max_intensity, columns )

    image = np.zeros( dimension )
    for idx in range(columns):
        image[row_coords[idx]][col_coords[idx]] = intensities[idx]

    if kernel is None:
        kernel = calculate_kernel( sigma=sigma, dimension=(row>>1, col>>1) )

    image = generate_random_coordinates( dimension=dimension, columns=columns, pixel_interval=pixel_interval, max_intensity=max_intensity )
    image = remove_boundary( image )
    image = convolution( image=image, kernel=kernel )

    if save_path is not None:
        imageio.imsave( save_path, image )

    return 255.0 * normalize( image )

def fake_images( images_to_generate, dimension=(512, 512), range_of_columns=(30, 70), sigmas=(9,100), max_intensity=5, pixel_interval=32 ):
    row, col = dimension
    #kernel = calculate_kernel( sigma=sigma, dimension=(row>>1, col>>1) )
    random_sigma = np.random.random_integers( *sigmas, images_to_generate )
    images = np.zeros( (images_to_generate, row, col) )
    for idx in range( images_to_generate ):
        images[idx,:,:] = fake_image( dimension=dimension, range_of_columns=range_of_columns, kernel=None, sigma=random_sigma[idx], max_intensity=max_intensity, offset=32, pixel_interval=pixel_interval, save_path=None )

    return images

def normalize( image ) :
    return (image-np.amin(image))/(np.amax(image)-np.amin(image)+1.0e-10)#scale to [0, 1]

def poisson( image, factor=10 ):
    image = np.random.poisson( image * factor ) / factor
    return image

def sim_noise( image, factor=0.75, poisson_factor=10, gaussian_var=0.1 ):
    sim = normalize( image )
    sim += np.random.random(image.shape) * factor
    sim -= np.random.random(image.shape) * factor
    sim += np.random.normal( 0.0, gaussian_var, sim.shape )
    sim = np.clip( sim, 0.0, 10.0 )
    sim = poisson( sim, poisson_factor )
    sim = np.clip( sim, 0.0, 1.0 )
    return sim

# simulate image with noise, and convolution
def simulate_median_image( image, kernel_size, use_noise, factor, poisson_factor, gaussian_var ):
    image = normalize( image )
    row, col = image.shape
    md = np.amax( image )
    sim = image
    if use_noise:
        sim = sim_noise( sim, factor=factor, poisson_factor=poisson_factor, gaussian_var=gaussian_var )
    sim = convolution( image=sim, kernel=np.ones((kernel_size, kernel_size)) )
    return sim

# simulate image with noise, and convolution
def simulate_image( image, sigma, use_noise, factor, poisson_factor, gaussian_var ):
    image = normalize( image )
    row, col = image.shape
    md = np.amax( image )
    sim = image
    if use_noise:
        sim = sim_noise( sim, factor=factor, poisson_factor=poisson_factor, gaussian_var=gaussian_var )
    kernel = np.zeros( (33, 33) )
    for r in range( 33 ):
        for c in range( 33 ):
            kernel[r][c] = exp( -((r-16)^2+(c-16)^2)/(sigma+sigma) )

    sim = convolution( image=sim, kernel=kernel )
    return sim

def apply_filter( padded_image, kernel_size ):
    kernel = np.ones( (kernel_size, kernel_size) )
    return convolution( image = padded_image, kernel = kernel )

def transform( image, use_noise, noisy_factor=None, poisson_factor=10, gaussian_var=0.1  ):
    row, col = image.shape
    filters_to_use =4
    padded_image =np.pad( image, ((64, 64), (64, 64)), 'symmetric')
    ans = np.zeros( (row, col, filters_to_use ) )

    if noisy_factor is None:
        noisy_factor = uniform( 0.25, 1.0 )

    for idx in range( filters_to_use ):
        if idx == 0 :
            procceed_image = simulate_image( image=padded_image, sigma=20, use_noise=use_noise, factor=noisy_factor, poisson_factor=poisson_factor, gaussian_var=gaussian_var )
        elif idx == 1 :
            procceed_image = simulate_image( image=padded_image, sigma=30, use_noise=use_noise, factor=noisy_factor, poisson_factor=poisson_factor, gaussian_var=gaussian_var )
        elif idx == 2 :
            procceed_image = simulate_median_image( image=padded_image, kernel_size=5, use_noise=use_noise, factor=noisy_factor, poisson_factor=poisson_factor, gaussian_var=gaussian_var )
        else :
            procceed_image = simulate_median_image( image=padded_image, kernel_size=7, use_noise=use_noise, factor=noisy_factor, poisson_factor=poisson_factor, gaussian_var=gaussian_var )
        procceed_image = procceed_image[64:64+row, 64:64+col]
        procceed_image = normalize( procceed_image ) * 255.0
        ans[:,:,idx] = procceed_image
    return ans

class DataLoader():
    def __init__(self, n_images=2048, img_res=(512, 512)):
        self.img_res = img_res
        self.n_images = n_images
        row, col = img_res
        self.batch_images = np.zeros( (self.n_images, row, col) )
        splitter = self.n_images // 2
        self.batch_images[:splitter, :, :] = gray(splitter, (row, col))
        self.batch_images[splitter:, :, :] = fake_images(self.n_images-splitter, (row, col), sigmas=(5,50), range_of_columns=(200, 400),max_intensity=1)
        np.random.shuffle( self.batch_images )
        self.n_batches = None
        self.counter = 0
        self.reset_counter = 1

        self.val_counter = 0
        self.validation_images = fake_images(self.n_images, (row, col), sigmas=(5,50), range_of_columns=(200, 400),max_intensity=5)

    def reset( self ):
        row, col = self.img_res
        self.reset_counter += 1
        splitter = self.n_images // (self.reset_counter+1)
        if reset_counter > 128:
            splitter = 0
        self.batch_images[:splitter, :, :] = gray(splitter, (row, col))
        self.batch_images[splitter:, :, :] = fake_images(self.n_images-splitter, (row, col), sigmas=(5,50), range_of_columns=(200, 400), max_intensity=self.reset_counter)
        np.random.shuffle( self.batch_images )
        self.counter = 0

    def load_batch(self, batch_size=3):
        if self.counter + batch_size >= self.n_images:
            self.counter = 0

        imgs_A, imgs_B = [], []

        for idx in range( self.counter, self.counter+batch_size ):
            img_A = self.batch_images[idx]
            img_B = transform( image=img_A, use_noise=True )

            imgs_A.append(img_A.reshape( (512, 512, 1)))
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/255.0
        imgs_B = np.array(imgs_B)/255.0
        self.counter += batch_size
        return imgs_B, imgs_A

    def load_validation_batch(self, batch_size=1 ):
        if self.val_counter + batch_size >= self.n_images:
            self.val_counter = 0

        imgs_A, imgs_B = [], []

        for idx in range( self.val_counter, self.val_counter+batch_size ):
            img_A = self.validation_images[idx]
            img_B = transform( image=img_A, use_noise=True )

            imgs_A.append(img_A.reshape(img_A.shape+(1,)))
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/255.0
        imgs_B = np.array(imgs_B)/255.0
        self.val_counter += batch_size
        return imgs_B, imgs_A

import tifffile
if __name__ == '__main__':
    N = 16
    loader = DataLoader(n_images=N)
    a, _ = loader.load_batch( N )
    b = []
    for idx in range( N ):
        b.append( sim_noise( np.squeeze(a[idx]) ) )
    b = np.asarray( b, dtype='float32' )
    a = np.asarray( normalize(a) * (256*256-1), dtype='uint16' )
    b = np.asarray( normalize(b) * (256*256-1), dtype='uint16' )
    tifffile.imsave( './a.tif', a )
    tifffile.imsave( './b.tif', b )


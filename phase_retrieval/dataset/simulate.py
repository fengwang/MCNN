import numpy as np
import imageio

def cal_k( scale, dim ):
    dk = 2.0 * np.pi / ( dim * scale )
    k1d = dk * ( -np.floor(dim/2.0) + np.arange(dim) )
    return np.fft.ifftshift(k1d)

def cal_kx_ky( scales=(0.2, 0.2), dims=(1024, 1024) ):
    return [ cal_k(scales[idx], dims[idx]) for idx in range(2) ]

def cal_kxm_kym( kx_ky ):
    return np.meshgrid(*kx_ky)

def cal_k2( kx_ky ):
    kxm, kym = cal_kxm_kym( kx_ky )
    return kxm**2 + kym**2

def cal_ka2( kx_ky, angle ):
    kxm, kym = cal_kxm_kym( kx_ky )
    ka1 = kxm * np.cos(angle) + kym * np.sin(angle)
    return ka1 * ka1

def cal_wnum( wlen, nref ):
    return 2.0 * np.pi / ( wlen * nref )

def cal_chi( wnum, ka2, k2 ):
    return np.sqrt(wnum**2-ka2) - 0.5 * (np.sqrt(wnum**2-k2)+wnum)

def cal_defocus_chi( wnum, k2 ):
    return (np.sqrt(wnum**2-k2)-wnum)

def cal_k2_aperture( NA, wnum, nref ):
    return (NA*wnum/nref)**2

def cal_aperture( k2, k2_aperture, fsmooth, dims=(1024, 1024) ):
    aperture = np.zeros( dims )
    aperture[k2 <= k2_aperture] = 1.0
    idx = (k2 > k2_aperture) * (k2 < (1.+fsmooth)*k2_aperture)
    aperture[idx] = 0.5 * (1. + np.cos(np.pi*(k2[idx]-k2_aperture) / (fsmooth*k2_aperture)))
    return aperture

def cal_ctf( chi, defoci, aperture ):
    return np.nan_to_num( np.exp(1.0j * chi * defoci) * aperture ).astype( 'complex64' )

def cal_envelope( k2, ka2, defoci, alpha ):
    return np.array( np.exp(-np.abs(0.5*k2-ka2)*(0.25*defoci*alpha)**2), dtype='float32' )

def cal_defocus_envelope( k2, defoci, alpha ):
    return np.array( np.exp(-k2*(0.5*defoci*alpha)**2), dtype='float32' )

def cal_image( phases, mx_range = 1.0, amplitude=None ):
    #amplitude = amplitude or np.ones( phases.shape )
    if amplitude is None:
        amplitude = np.ones( phases.shape )
    return np.sqrt(amplitude) * np.exp( mx_range * 1.0j * phases )

def cal_image_padding( image, mode='reflect', padding=None ):
    if mode == 'none':
        return image
    padding = padding or [dim >> 1 for dim in phases.shape]
    return np.pad(image, (padding, padding), mode=mode)

def cal_wave( ctf, image ):
    return ctf * np.fft.fft2( image )

def cal_astig( wave, envelope ):
    fsdata = np.abs(np.fft.ifft2(wave))**2
    return np.asarray( np.fft.ifft2(np.fft.fft2(fsdata)*envelope), dtype='float32' )

def cal_astigmatism( phases, alpha, NA, wlen, nref, defoci, angle, scales, max_range, padding, mode, amplitude, fsmooth ):
    image = cal_image_padding( cal_image( phases, max_range, amplitude ), mode, padding )
    kx_ky = cal_kx_ky( scales=scales, dims=image.shape )
    k2 = cal_k2( kx_ky )
    ka2 = cal_ka2( kx_ky, angle )
    wnum = cal_wnum( wlen, nref )
    k2_aperture = cal_k2_aperture( NA, wnum, nref )
    aperture = cal_aperture( k2, k2_aperture, fsmooth, image.shape )
    chi = cal_chi( wnum, ka2, k2 )
    ctf = cal_ctf( chi, defoci, aperture )
    wave = cal_wave( ctf, image )
    envelope = cal_envelope( k2, ka2, defoci, alpha )
    astig = cal_astig( wave, envelope )
    if mode == 'none':
        return astig
    return astig[padding[0]:padding[0]+phases.shape[0], padding[1]:padding[1]+phases.shape[1]]

def cal_defocus( phases, alpha, NA, wlen, nref, defoci, scales, max_range, padding, mode, amplitude, fsmooth ):
    image = cal_image_padding( cal_image( phases, max_range, amplitude ), mode, padding )
    kx_ky = cal_kx_ky( scales=scales, dims=image.shape )
    k2 = cal_k2( kx_ky )
    #ka2 = cal_ka2( kx_ky, angle )
    wnum = cal_wnum( wlen, nref )
    k2_aperture = cal_k2_aperture( NA, wnum, nref )
    aperture = cal_aperture( k2, k2_aperture, fsmooth, image.shape )
    chi = cal_defocus_chi( wnum, k2 )
    ctf = cal_ctf( chi, defoci, aperture )
    wave = cal_wave( ctf, image )
    #envelope = cal_envelope( k2, ka2, defoci, alpha )
    envelope = cal_defocus_envelope( k2, defoci, alpha )
    astig = cal_astig( wave, envelope )
    if mode == 'none':
        return astig
    return astig[padding[0]:padding[0]+phases.shape[0], padding[1]:padding[1]+phases.shape[1]]

if __name__ == '__main__':
    wave_length = 0.525
    #refractive_index = 1.0
    refractive_index = 1.518
    #numerical_aperture = 0.65
    numerical_aperture = 1.35
    fsmooth = 2.0
    #astigmatic_defocus = 150.0
    astigmatic_defocus = 1.0
    alpha = 0.05
    maximum_phase_range = 1.0
    # phase_image_path = './orig.png'
    #phase_image_path = './orig.jpg'
    #phase_image_path = './999.gray.png'
    phase_image_path = './0000000350.jpg'
    #scales = (0.2, 0.2)
    scales = (0.156250, 0.156250)
    padding_mode = 'reflect'
    #rotation_angle = 2.19039955
    rotation_angle = 0.19039955
    phases = np.array( imageio.imread( phase_image_path ), dtype='float32' )
    phases = ( phases - np.amin(phases) ) / ( np.amax(phases) - np.amin(phases) + 1.0e-10 )
    amplitude = phases * 0.8 + 0.2
    #astig = cal_astigmatism( phases, alpha, numerical_aperture, wave_length, refractive_index, astigmatic_defocus, rotation_angle, scales, maximum_phase_range, [dim>>1 for dim in phases.shape], padding_mode, amplitude, fsmooth )
    astig = cal_astigmatism( phases, alpha, numerical_aperture, wave_length, refractive_index, astigmatic_defocus, rotation_angle, scales, maximum_phase_range, [dim>>1 for dim in phases.shape], 'none', amplitude, fsmooth )
    defocus = cal_defocus( phases, alpha, numerical_aperture, wave_length, refractive_index, astigmatic_defocus, scales, maximum_phase_range, [dim>>1 for dim in phases.shape], 'none', amplitude, fsmooth )
    imageio.imsave( './astig.png', astig )
    imageio.imsave( './defocus_1.png', defocus )
    astigmatic_defocus = -1.0
    defocus = cal_defocus( phases, alpha, numerical_aperture, wave_length, refractive_index, astigmatic_defocus, scales, maximum_phase_range, [dim>>1 for dim in phases.shape], 'none', amplitude, fsmooth )
    imageio.imsave( './defocus_-1.png', defocus )
    astigmatic_defocus = -6.1263185
    defocus = cal_defocus( phases, alpha, numerical_aperture, wave_length, refractive_index, astigmatic_defocus, scales, maximum_phase_range, [dim>>1 for dim in phases.shape], 'none', amplitude, fsmooth )
    imageio.imsave( './defocus_-6.png', defocus )
    astigmatic_defocus = 10.282856
    defocus = cal_defocus( phases, alpha, numerical_aperture, wave_length, refractive_index, astigmatic_defocus, scales, maximum_phase_range, [dim>>1 for dim in phases.shape], 'none', amplitude, fsmooth )
    imageio.imsave( './defocus_10.png', defocus )

    def save( path, img ):
        img = np.asarray( img / np.amax(img) * 255.0, dtype='uint8' )
        imageio.imsave( path, img )

    for depth in [ -48.62, -37.5, -28.9, -17.2, -10.0, -6.0, -1.0, 1.0, 6.0, 10.0, 17.2, 28.9, 37.5, 48.62 ]:
        astigmatic_defocus = depth
        defocus = cal_defocus( phases, alpha, numerical_aperture, wave_length, refractive_index, astigmatic_defocus, scales, maximum_phase_range, [dim>>1 for dim in phases.shape], 'none', amplitude, fsmooth )
        save( f'./defocus_f{depth}.png', defocus )





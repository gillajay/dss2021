import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_image(image,
               cmap='gray',
               title='Image',
               input_ratio=None,
               interval_type='zscale',
               stretch='linear',
               percentile=99,
               xlim_minus=0,
               xlim_plus=1000,
               ylim_minus=0,
               ylim_plus=1000):
    
    from astropy.visualization import (ZScaleInterval,
                                       PercentileInterval,
                                       MinMaxInterval,
                                       ImageNormalize,
                                       simple_norm)
        
    fig = plt.figure(figsize=(10, 10))
    
    ax = fig.add_subplot(1, 1, 1)
    
    if interval_type== 'zscale': 
        norm = ImageNormalize(image, interval=ZScaleInterval())
    
    elif interval_type == 'percentile':
        norm = ImageNormalize(image, 
                              interval=PercentileInterval(percentile))
    elif interval_type == 'minmax':
        norm = ImageNormalize(image,
                             interval=MinMaxInterval())
    elif interval_type == 'simple_norm':
        norm = simple_norm(image, stretch)
        
    im = ax.imshow(image[xlim_minus:xlim_plus,
                         ylim_minus:ylim_plus],
                   cmap=cmap,
                   interpolation='none',
                   origin='lower',
                   norm=norm)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)

    plt.colorbar(im, cax=cax, label='ADU')
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("y [pixels]")
    ax.set_title(title, fontsize=14)
        
    plt.show()
    
def get_read_noise(image,
                   read_noise_std,
                   gain=1):
    """
    Generate read noise 2D array.
    
    Parameters
    input:
    image: numpy array
        Science image whose shape read noise array will match
    read_noise_std: float
        Read noise of the camera [electrons rms]
    gain: float (optional)
        Gain of the camera, [electrons/ADU]
    output:
        numpy array with read noise who shape matches input science 
        array
    """
    array_shape = image.shape
    read_noise_array = np.random.normal(loc=0,
                                        scale=read_noise_std/gain,
                                        size=array_shape)
    return read_noise_array

# Noise generation functions
def get_bias_level(image,
                   bias_value,
                   add_bad_columns=True,
                   num_columns=5):
    """
    Generate simulated bias 2D array.
    
    Parameters
    ----------
    image: numpy array
         Science image whose shape read noise array will match
    bias_value: float
         Bias value to add to the image [ADU]
    add_bad_columns: bool, optional
         Optional argument to add bad columns to the bias image. The
         bad columns will have higher bias level than the other pixels.
    num_columns: float
         Number of bad columns to add. Default is 5.
    output:
        numpy array with the bias level with shape matches input science 
        array
    """
    bias_im = np.zeros_like(image) + bias_value
    
    if add_bad_columns:
        rng = np.random.RandomState(seed=24) 

        columns = rng.randint(1, image.shape[0], size=num_columns)
        col_value = rng.randint(int(0.3*bias_value), 
                                int(0.8*bias_value), 
                                size=image.shape[1])
        
        # Add additional brightness to random columns
        for column in columns:
            bias_im[:, column] = bias_value + col_value
    return bias_im

def get_dark_current(image, 
                     dark_current,
                     exposure_time,
                     gain=1.0,
                     hot_pixels=False,
                     hot_pixels_percentage=0.1):
    """
    Generate simulated dark current 2D array.
    
    Parameters
    ----------
    image: numpy array
         Science image whose shape read noise array will match
    dark_current: float
         Dark current level [electrons/sec/pixel]
    exposure_time: float
         Exposure time of the image [seconds]
    gain: float, optional
         Camera gain [electron/ADU]
    hot_pixels: bool, optional
         Whether or not to add hot pixels to the image []
    hot_pixels_percentage: float, optional
         Percentage of hot pixels []
    output:
        numpy array with the dark current with shape matches the
        input science array
    """
    # get dark current level in ADU
    dark_adu = dark_current * exposure_time / gain
    
    # create a 2D sampled from a Poisson distribution
    dark_im = np.random.poisson(lam=dark_adu, size=image.shape)
    
    if hot_pixels:
        y_max, x_max = dark_im.shape
        n_hot = int((hot_pixels_percentage/100) * x_max * y_max)
        
        rng = np.random.RandomState(100)
        hot_x = rng.randint(0, x_max, size=n_hot)
        hot_y = rng.randint(0, y_max, size=n_hot)
        
        # set hot pixels to be 1000 times the dark current level
        hot_current = 100 * dark_current
        
        for i in range(n_hot):
            dark_im[hot_x[i], hot_y[i]] = (hot_current  
                                           * exposure_time / gain)
    return dark_im

def get_sky_bkg(image, 
                sky_noise_electrons,
                exposure_time,
                gain=1.0):    
    """
    Generate simulated sky background 2D array.
    
    Parameters
    ----------
    image: numpy array
         Science image whose shape read noise array will match
    sky_noise_electrons: float
         Sky background level [electrons/sec/pixel]
    exposure_time: float
         Exposure time of the image [seconds]
    gain: float, optional
         Camera gain [electron/ADU]
    output:
        numpy array with the sky background with shape matches the
        input science array
    """
    sky_adu = sky_noise_electrons * exposure_time / gain
    sky_im = np.random.poisson(lam=sky_adu, size=image.shape)
    return sky_im

def get_flat(image,
             percent_variations=5):
    """
    Generate a flat field image.
    
    Parameters
    ----------
    image: numpy array
         Science image whose shape read noise array will match
    percent_variations: float, optional
         Maximum percentage variation in pixel-to-pixel
         sensitivity []
    output:
        numpy array with the flat field with a shape that matches the
        input science array
    """
    # create a random seed to generate the same random image each time
    np.random.seed(0)
    
    # sample from a uniform distribution
    flat = np.random.uniform(low=1-(percent_variations/100),
                             high=1.0,
                             size=image.shape)
    return flat

def abmag_to_fnu(abmag):
    """
    Convert ABmag to flux density [erg/s/cm^2/Hz] 
    """
    return 10**((abmag+48.6) / -2.5)

def fnu_to_abmag(fd_freq):
    """
    Convert flux density [erg/s/cm^2/Hz] to ABmag
    """
    return -2.5*np.log10(fd_freq) - 48.60

def noise_image_interactive(bias_value=200,
                            read_noise_std=10,
                            dark_current=0.1,
                            hot_pixels=False,
                            hot_pixels_percentage=0.1,
                            sky_noise_electrons=0.5,
                            exposure_time=100,
                            percent_variations=5,
                            num_bad_columns=5,
                            plot_scale='linear'):
    
    gain = 0.76 # electron/ADU
    
    # Construct a blank image
    blank_image = np.zeros([1000, 1000])
    
        
    # Construct a bias only image
    im_bias = get_bias_level(image=blank_image,
                         bias_value=bias_value,
                         num_columns=num_bad_columns)
    
    # Construct a read noise only image
    im_read_noise = get_read_noise(image=blank_image,
                                   read_noise_std=read_noise_std,
                                   gain=gain)
    
    # Construct a dark current only image
    im_dark_current = get_dark_current(image=blank_image,
                               dark_current=dark_current,
                               gain=gain,
                               exposure_time=exposure_time,
                               hot_pixels=hot_pixels,
                               hot_pixels_percentage=hot_pixels_percentage)
    
    # Construct a sky noise only image
    im_sky_noise = get_sky_bkg(image=blank_image,
                               sky_noise_electrons=sky_noise_electrons,
                               gain=gain,
                               exposure_time=exposure_time)
    
    # Construct the flat field variations 
    im_flat = get_flat(image=blank_image,
                       percent_variations=percent_variations)

    # Flat field correct the sky noise image
    im_sky_noise_corr = np.multiply(im_sky_noise, im_flat)


    noise = (im_read_noise
           + im_dark_current
           + im_sky_noise_corr
           + im_bias)
    
        # plots
    if plot_scale == 'zscale':
        interval_type = 'zscale'
        stretch = 'linear'
    elif plot_scale == 'linear':
        interval_type = 'simple_norm'
        stretch = plot_scale
    elif plot_scale == 'log':
        interval_type = 'simple_norm'
        stretch = plot_scale 
    elif plot_scale == 'sqrt':
        interval_type = 'simple_norm'
        stretch = plot_scale
    
    # plot
    plot_image(image=noise, 
               title="Noise image",
               interval_type=interval_type,
               stretch=stretch)
    
    return None
    
def get_SNR(source_count_rate,
            read_noise_std,
            sky_noise_electrons,
            dark_current,
            exposure_time,
            n_exp=1,
            n_pix=1):    
    """
    Simple CCD equation. Given source count rate, exposure time, 
    read and sky noise, calculates the SNR.
    inputs:
    1) source_count_rate: (float) [electrons/second]
    2) read_noise_std: (float) [electrons/pixel]
    3) sky_noise_electrons: (float) [electrons/second/pixel]
    4) dark_current: (float) [electrons/second/pixel]
    5) exposure_time: (float) exposure time of sensor [seconds]
    7) n_exp: (int) number of exposures 
    returns:
    1) signal-to-noise ratio: (float)
    """
    signal = source_count_rate * exposure_time * n_exp
    
    # invidivual noise terms
    noise_source = source_count_rate * exposure_time * n_exp
    noise_sky = sky_noise_electrons * exposure_time * n_exp * n_pix
    noise_dark = dark_current * exposure_time * n_exp * n_pix
    noise_read = n_exp * read_noise_std**2 * n_pix
    
    noise = np.sqrt(noise_source 
                  + noise_sky
                  + noise_dark
                  + noise_read)
    return signal / noise
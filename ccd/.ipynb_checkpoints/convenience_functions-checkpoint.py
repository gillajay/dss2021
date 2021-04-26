import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.constants import c, h
from astropy import units as u
plt.rcParams.update({'font.size': 16})
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_image(image,
               cmap='gray',
               title='Image',
               input_ratio=None,
               interval_type='zscale',
               percentile=99,
               default_ticks=False):
    from astropy.visualization import (ZScaleInterval,
                                       PercentileInterval,
                                       MinMaxInterval,
                                       ImageNormalize)
        
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

    im = ax.imshow(image,
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
    
    if default_ticks == False:
        ax.set_xticks(np.arange(0, 1200, 200))
        ax.set_yticks(np.arange(0, 1200, 200))
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

def get_bias_level(image,
               bias_value,
               add_bad_columns=True,
               num_columns=5):
    """
    Generate simulated bias 2D array.
    
    Parameters
    ----------
    input: numpy array
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
        rng = np.random.RandomState(seed=23) 

        columns = rng.randint(0, image.shape[1], size=num_columns)
        col_value = rng.randint(0, int(0.4*bias_value), size=image.shape[0])
        
        # Add additional brightness to random columns
        for column in columns:
            bias_im[:, column] = bias_value + col_value
    return bias_im

def get_dark_current(image, 
                dark_current,
                exposure_time,
                gain=1.0,
                hot_pixels=False,
                hot_pixels_percentage=0.01):
    dark_adu = dark_current * exposure_time / gain
    dark_im = np.random.poisson(lam=dark_adu, size=image.shape)
    
    if hot_pixels:
        y_max, x_max = dark_im.shape
        n_hot = int((hot_pixels_percentage/100) * x_max * y_max)
        
        rng = np.random.RandomState(100)
        hot_x = rng.randint(0, x_max, size=n_hot)
        hot_y = rng.randint(0, y_max, size=n_hot)
        
        hot_current = 1000 * dark_current
        
        for i in range(len(hot_x)):
            dark_im[hot_x[i], hot_y[i]] = (hot_current  
                                           * exposure_time / gain)
    return dark_im

def get_sky_bkg(image, 
                sky_noise_electrons,
                exposure_time,
                gain=1.0):    
    sky_adu = sky_noise_electrons * exposure_time / gain
    sky_im = np.random.poisson(lam=sky_adu, size=image.shape)
    return sky_im

def flambda_from_fnu(bandpass, 
                     fnu,
                     frequencies):
    """
    Converts frequency dependent flux density to wavelength 
    dependent flux density
    
    inputs:
    bandpass: (array)
        instrument bandpass 
    fnu: (array)
        wavelength dependent flux density n 
        increments of 1 nm [erg/s/cm^2/Hz] 
    frequencies: (array)
        frequencies  
    band: (str)
    returns:
        (float)
        mean of flux density in [erg/s/cm^2/nm]
    """
    bandpass = np.asarray(bandpass)
    frequencies = np.asarray(frequencies)
    frequencies *= u.Hz
        
    fnu *= (u.erg/u.s/u.cm**2/u.Hz)
    
    del_nu =((np.max(frequencies) - np.min(frequencies)) 
             / len(frequencies))
    
    numerator = np.trapz(bandpass*fnu,
                         x=frequencies,
                         dx=del_nu)
    denominator = np.trapz(y=bandpass*c/frequencies**2.,
                         x=frequencies,
                         dx=del_nu)
    flambda = (numerator / denominator).to(u.erg/u.s/u.cm**2/u.nm).value
    
    return flambda

def countrate_from_flambda(flambda,
                           bandpass,
                           wavelengths,
                           illumination_area):    
    
    flambda *= u.erg/u.s/u.cm**2/u.nm
    wave_int = np.asarray(wavelengths)
    wave_int *= u.nm
    illumination_area *= u.cm**2.
    bandpass = np.asarray(bandpass)
    
    integrand = (illumination_area  
               * flambda 
               * bandpass 
               * wave_int) / (h * c)

    del_lam = ((np.max(wave_int) - np.min(wave_int)) / len(wave_int))

    count_rate = (np.trapz(y=integrand, 
                           x=wave_int, 
                           dx=del_lam).to(u.s**-1)) # electrons/s
    
    return count_rate.value

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

def wave_to_freq(wavelength):
    """
    Wavelengeth to frequency converter
    """
    return (c / (wavelength*u.nm)).to(u.Hz).value
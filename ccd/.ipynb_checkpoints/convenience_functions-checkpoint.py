import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from astropy.io import fits
plt.rcParams.update({'font.size': 16})
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_image(image,
               cmap='gray',
               title='Image',
               input_ratio=None,
               interval_type='zscale',
               percentile=99):
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
    ax.set_xticks(np.arange(0, 1200, 200))
    ax.set_yticks(np.arange(0, 1200, 200))
    plt.show()
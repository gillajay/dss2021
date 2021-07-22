Dunlap Summer School 2021 Lab Activity 1 (CCD Data Reduction and Photometry).

**Instructions**
1) Lab Activity 1 can be completed using two ways:

a) Downloading the required Jupyter notebooks and associated data files onto your local machine.

b) Using the provided binder links (which hosts the notebooks online).

2) For path (a): the download link for Lab Activity 1 will be provided in the Slack channel.
3) For those who do not have Jupyter installed, we recommend installing Anaconda (see instructions here: https://docs.anaconda.com/anaconda/install/)
4) Lab Activity 1 requires the following packages: numpy==1.19.2, matplotlib==3.3.4, ipywidgets==7.6.3, pandas==1.2.3, astropy==4.2

which can be installed via `pip install <package name>` in terminal if not already installed.  

5) For path (b): the binder link for Lab Activity 1 will be provided in the Slack channel.

**Lab Activity 1 Overview**
1) Lab Activity 1 consists of ccd data reduction and photometric calibration exercises.
2) The exercises are in Jupyter notebook format and are located in the `ccd_exercises/exercises` folder.
3) The learning outcomes for the different Jupyter notebooks are listed below.

4) **Notebook 1: Basics of CCD operation**
    - We will learn about the basics of CCD operation.
5) **Notebook 2:  Realistic Image Simulations**
    - We will learn about the different noise components that make up a CCD image:
        - dark current
        - read noise
        - sky noise
        - pixel-by-pixel sensitivity variations
        - fixed pattern noise and bias level
    - We will learn how all these different components combine in a real image.
    - We will study the effect of interactively changing different noise components on the image.
6) **Notebook 3: Signal-to-Noise Ratio**
    - We will learn about what the signal-to-noise ratio (SNR) is.
    - We will learn about the different components that lead to the calculation of the SNR.
    - We will study the impact of changing the exposure time of your observation on the SNR.
    - We will study the how the SNR changes for one long exposure compared to multiple short exposures.
    - We will interactively vary the parameters in the SNR calculation, to see which parameters dominate.
7) **Notebook 4: Photometric Calibration of Realistic Image**
    - We will learn about astronomical magnitude systems and the flux densities (or spectral energy distributions) of a star.
    - We will learn about how to construct a bandpass of the instrument by combining multiple instrument components such as:
        - The telescope throughput
        - The camera quantum efficiency
        - The filter response
    - We will then use realistic images of a g2v spectral type star observed by our instrument.
    - We will clean the images using calibration images.
    - We will then estimate the sky background level in digital units.
    - Knowing the spectral type of the star and the observed count rate, we will estimate the sensitivity of the instrument in different bands.
    - Knowing the sensitivity, we will estimate the AB magnitude of the star as well as the night sky background per square arcsecond in the different bands.










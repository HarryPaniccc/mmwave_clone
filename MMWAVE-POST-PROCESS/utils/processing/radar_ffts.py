import numpy as np
# from numpy.fft import fft,fftshift
from scipy.fft import fft, fftshift


def azimuth_fft(data, azimuth_pad, channel_select=[0,1,2,3,8,9,10,11] , azimuth_window=np.hanning):
    """Performs the range azimuth process on a radar data cube after ard with a given range zero pad and window function, channel zero pad and window functin.
    It does a 1D FFT across the channel/angle/3rd axis) .

        input: data             -> must be a 3D radar data cube with the dimensions [numsamples,numChirps,numChannels]
        input: azimuth_pad      -> numbber of samples to zero pad channel/angle axis
        input: azimuth_window   -> window function handle used for azimuth fft. Default is numpy hanning window.

        output: rd_fft          -> 3D matrix of size (nSamples+range_pad, nChirps+dopple_pad, nVChannels). This is complex data. Not 20*log().
        
    """

    # window channel dimension
    nVChannels = len(channel_select)
    window = np.tile(azimuth_window(nVChannels)[np.newaxis, np.newaxis, :], (data.shape[0], data.shape[1], 1))

    
    # doppler fft
    return fftshift(fft(data[:,:,channel_select]*window,nVChannels + azimuth_pad ,2),2)


def azimuth_sum(range_azimuth_fft):
    """Converts doppler cube to single map in dB. Use this function before giving map to CFAR. Sums along the unused chirp axis.

        input: data             -> must be a 3D array with the dimensions [numsamples,numChirps,numChannels]

        output: map          -> 2D matrix of size (nSamples+range_pad, nChirps+dopple_pad).
        
    """

    range_azimuth_fft = np.sum(np.abs(range_azimuth_fft),1)
    return 20*np.log10((range_azimuth_fft)/np.max(np.max(np.abs(range_azimuth_fft))))

def range_doppler_fft(data, range_pad, doppler_pad, range_window=np.hanning,doppler_window=np.hanning):
    """Performs the range doppler process on a radar data cube with a given range zero pad and window function, doppler zero pad and window functin.
    It does a 1D range fft on each chirp in each channel and then a 1D doppler fft accross chirps on each sample to get the range doppler map. It
    does this for each virtual channel.

        input: data             -> must be a 3D array with the dimensions [numsamples,numChirps,numChannels]
        input: range_pad         -> numbber of samples to zero pad sample/range axis
        input: doppler_pad       -> numbber of samples to zero pad chirp/doppler axis
        input: range_window     -> window function handle used for range axis. Default is numpy hanning window.
        input: doppler_window   -> window function handle used for chirp axis. Default is numpy hanning window.

        output: rd_fft          -> 3D matrix of size (nSamples+range_pad, nChirps+dopple_pad, nVChannels). This is complex data. Not 20*log().
        
    """
    # range fft
    data = range_fft(data, range_pad, range_window)

    # make a 3D window 
    window = np.tile(doppler_window(data.shape[1])[np.newaxis, :, np.newaxis], (data.shape[0], 1, data.shape[2]))

    # doppler fft
    return fftshift(fft(data*window,data.shape[1] + doppler_pad ,1),1)


def range_fft(data, range_pad, range_window=np.hanning):
    """Performs the range fft on a radar data cube with a given range zero pad and window function.
    It does a 1D range fft on each chirp for the channel specified.

        input: data             -> must be a 3D array with the dimensions [numsamples,numChirps,numChannels]
        input: range_pad         -> numbber of samples to zero pad sample/range axis
        input: range_window     -> window function handle used for range axis. Default is numpy np.hanning.

        output: r_fft          -> 3D matrix of size (nSamples+range_pad, nChirps, nVChannels). This is complex data. Not 20*log(). Each of page of this
                                  matrix is a set of range_ffts
        
    """

    # make a 3D window 
    window = np.tile(range_window(data.shape[0])[:, np.newaxis, np.newaxis], (1, data.shape[1], data.shape[2]))

    # range fft
    return fft(data*window,data.shape[0] + range_pad ,0)



def range_doppler_sum(range_doppler_fft):
    """Converts rd doppler cube to single range doppler map in dB. Use this function before giving CFAR. 

        input: data             -> must be a 3D array with the dimensions [numsamples,numChirps,numChannels]
        input: range_pad         -> numbber of samples to zero pad sample/range axis
        input: doppler_pad       -> numbber of samples to zero pad chirp/doppler axis

        output: rd_map          -> 2D matrix of size (nSamples+range_pad, nChirps+dopple_pad).
        
    """

    range_doppler_fft = np.sum(np.abs(range_doppler_fft),2)
    return 20*np.log10(np.abs(range_doppler_fft)/np.max(np.max(np.abs(range_doppler_fft))))



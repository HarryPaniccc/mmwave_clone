import numpy as np
from scipy.signal import sosfiltfilt, butter, cheby1,cheby2,ellip, bessel

def filter_butter(signal, ftype, crtical_fs , order, sample_time):
    """ Created a zero phase filter of the chosen type with desired frequencies:
    input: signal -> the time series to filter
    input: ftype  -> the type of filter. {"lowpass", "highpass", "bandpass", "bandstop"}
    input: critcal_fs -> scalar critcal frequency for low and highpass and a tuple for bandpass and band stop
    input: order ->  order of the filter. order = 1,2,3,4 etc.
    input: sample_time -> the time between samples

    output: filtered_signal -> the signal filtered according to specifications
    """
    
    sample_rate = 1/sample_time
    sos = butter(order, crtical_fs, btype=ftype, analog=False, output='sos', fs=sample_rate)
    signal = np.pad(signal,(1000000,1000000),"edge") 
    filtered_signal = sosfiltfilt(sos, signal)
    filtered_signal = filtered_signal[1000000:-1000000]
    sos = butter(order*2, crtical_fs, btype=ftype, analog=False, output='sos', fs=sample_rate)

    return filtered_signal, sos

def filter_cheby1(signal, ftype, crtical_fs , order, sample_time, ripple = 1):
    """ Created a zero phase filter of the chosen type with desired frequencies:
    input: signal -> the time series to filter
    input: ftype  -> the type of filter. {"lowpass", "highpass", "bandpass", "bandstop"}
    input: critcal_fs -> scalar critcal frequency for low and highpass and a tuple for bandpass and band stop
    input: order ->  order of the filter. order = 1,2,3,4 etc.
    input: sample_time -> the time between samples

    output: filtered_signal -> the signal filtered according to specifications
    """
    
    sample_rate = 1/sample_time
    sos = cheby1(order,ripple, crtical_fs, btype=ftype, analog=False, output='sos', fs=sample_rate)
    signal = np.pad(signal,(100000,100000),"edge") 
    filtered_signal = sosfiltfilt(sos, signal)
    filtered_signal = filtered_signal[100000:-100000]
    sos = cheby1(order*2,ripple, crtical_fs, btype=ftype, analog=False, output='sos', fs=sample_rate)

    return filtered_signal, sos

def filter_cheby2(signal, ftype, crtical_fs , order, sample_time, ripple = 1):
    """ Created a zero phase filter of the chosen type with desired frequencies:
    input: signal -> the time series to filter
    input: ftype  -> the type of filter. {"lowpass", "highpass", "bandpass", "bandstop"}
    input: critcal_fs -> scalar critcal frequency for low and highpass and a tuple for bandpass and band stop
    input: order ->  order of the filter. order = 1,2,3,4 etc.
    input: sample_time -> the time between samples

    output: filtered_signal -> the signal filtered according to specifications
    """
    
    sample_rate = 1/sample_time
    sos = cheby2(order,ripple, crtical_fs, btype=ftype, analog=False, output='sos', fs=sample_rate)
    signal = np.pad(signal,(100000,100000),"edge") 
    filtered_signal = sosfiltfilt(sos, signal)
    filtered_signal = filtered_signal[100000:-100000]
    sos = cheby2(order*2,ripple, crtical_fs, btype=ftype, analog=False, output='sos', fs=sample_rate)

    return filtered_signal, sos

def filter_ellip(signal, ftype, crtical_fs , order, sample_time, ripple=1,stop_band_attenuation=80):
    """ Created a zero phase filter of the chosen type with desired frequencies:
    input: signal -> the time series to filter
    input: ftype  -> the type of filter. {"lowpass", "highpass", "bandpass", "bandstop"}
    input: critcal_fs -> scalar critcal frequency for low and highpass and a tuple for bandpass and band stop
    input: order ->  order of the filter. order = 1,2,3,4 etc.
    input: sample_time -> the time between samples

    output: filtered_signal -> the signal filtered according to specifications
    """
    
    sample_rate = 1/sample_time
    sos = ellip(order,ripple,stop_band_attenuation, crtical_fs, btype=ftype, analog=False, output='sos', fs=sample_rate)
    signal = np.pad(signal,(100000,100000),"edge") 
    filtered_signal = sosfiltfilt(sos, signal)
    filtered_signal = filtered_signal[100000:-100000]
    sos = ellip(order*2,ripple,stop_band_attenuation, crtical_fs, btype=ftype, analog=False, output='sos', fs=sample_rate)

    return filtered_signal, sos

def filter_bessel(signal, ftype, crtical_fs , order, sample_time):
    """ Created a zero phase filter of the chosen type with desired frequencies:
    input: signal -> the time series to filter
    input: ftype  -> the type of filter. {"lowpass", "highpass", "bandpass", "bandstop"}
    input: critcal_fs -> scalar critcal frequency for low and highpass and a tuple for bandpass and band stop
    input: order ->  order of the filter. order = 1,2,3,4 etc.
    input: sample_time -> the time between samples

    output: filtered_signal -> the signal filtered according to specifications
    """
    
    sample_rate = 1/sample_time
    sos = bessel(order, crtical_fs, btype=ftype, analog=False, output='sos', fs=sample_rate)
    
    signal = np.pad(signal,(100000,100000),"edge") 
    filtered_signal = sosfiltfilt(sos, signal)
    filtered_signal = filtered_signal[100000:-100000]

    sos = bessel(order*2, crtical_fs, btype=ftype, analog=False, output='sos', fs=sample_rate)

    return filtered_signal, sos



def moving_average_filter(signal,window_size):

    output = np.zeros(len(signal))
    padded_signal = np.pad(signal,(window_size//2,window_size//2),"edge")
    start_index = window_size//2
    for i in range(len(output)):
        start = int(i+start_index-window_size//2)
        end = int(i+start_index+window_size//2)

        output[i] = np.sum(padded_signal[start:end])/window_size

    return output        

from scipy.signal import hilbert
import cv2 as cv
def amplitude_threshold_filter(time,signal):
    # filter bank
    bandwidth = 0.2 # Hz 
    start_frequency = 1
    end_frequency = 3
    l_bounds = np.arange(start_frequency,end_frequency,bandwidth)
    u_bounds = l_bounds + bandwidth
    
    # 2D Filter
    i = 0
    filter_image = np.zeros((len(l_bounds),len(signal)))
    image_envelope = np.zeros((len(l_bounds),len(signal)))
    for lc, uc in zip(l_bounds,u_bounds):
        
        notch_signal, _ = filter_butter(signal,"bandpass",(lc,uc),4,np.mean(np.diff(time)))
        filter_image[i,:] = notch_signal
        
        analytic_signal = hilbert(notch_signal)
        image_envelope[i,:] = np.abs(analytic_signal)
        
        i+=1      
    
    # threshold
    threshold_map = np.ones(filter_image.shape)
    threshold_map[image_envelope>0.0004] = 0
    threshold_map[image_envelope<0.00005] = 0
    filter_gain = cv.GaussianBlur(threshold_map, (25,3),0)
    filter_out_2D = filter_image*filter_gain
    filter_out = np.sum(filter_out_2D,0)
    
    return filter_out
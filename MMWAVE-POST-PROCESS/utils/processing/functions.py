import numpy as np
from matplotlib import pyplot as plt 
from scipy.signal import sosfiltfilt, butter

def range_bins_to_ranges(bin_numbers, number_of_samples=96, zero_padding=400, max_range=7.5):
    distances = np.flip(np.linspace(0,max_range,(number_of_samples + zero_padding)))
    ranges = []
    for bin_number in bin_numbers:
        ranges.append(distances[int(bin_number)])
        
    return np.array(ranges)

def root_mean_square_error(predictions, expected):
    return np.sqrt(np.mean((predictions-expected)**2))

def percentage_accuracy(predictions, expected):
    return np.abs(100 - 100*np.abs(expected-predictions)/expected)

def integrate(signal, dt):
    
    integrated_signal = np.zeros(len(signal))
    for i in range(len(signal)):
        integrated_signal[i] = np.sum(signal[:i+1])*dt

    return integrated_signal
    
def envelope_detector(time_vals,signal_vals,lpf_cutoff):

    abs_sig = np.abs(signal_vals)

    sos = butter(4, lpf_cutoff, btype="lowpass", analog=False, output='sos', fs=1/np.mean(np.diff(time_vals)))
    signal = np.pad(abs_sig,(1000000,1000000),"edge") 
    filtered_signal = sosfiltfilt(sos, signal)
    filtered_signal = filtered_signal[1000000:-1000000]
    envelope = filtered_signal
    return envelope

def get_freqeunecy_content(time_vals,input_signal,padding,bpm=True, normalise=True,
                           window_func = np.hanning):
    FFT_points = len(input_signal)+padding 
    sample_period = np.mean(np.diff(time_vals))

    fft = np.fft.fftshift(np.fft.fft(input_signal*window_func(len(input_signal)),FFT_points))
    freq_phase = np.angle(fft)

    if normalise:
        freq_mag=20*np.log10(np.abs(fft)/np.max(np.max(np.abs(fft))))

    else:
        freq_mag=20*np.log10(np.abs(fft))
        

    if not bpm:
        freqs = np.fft.fftshift(np.fft.fftfreq(n=FFT_points, d=sample_period))

    else:
        freqs = np.fft.fftshift(np.fft.fftfreq(n=FFT_points, d=sample_period))*60

    return freqs, freq_mag, freq_phase

def plot_frequency_content(time_vals,input_signal,padding,sample_period,
                           frequency_bound, magnitude_bound, source,
                           expected_rate = None, bpm=True, normalise=False,
                           window_func = np.hanning):
    """Examine frequnecy content to look at where to remove quantization noise"""
    FFT_points = len(input_signal)+padding + 1
    plot_handles = []
    fig = plt.figure( figsize=(8, 6))
    ax = fig.add_subplot(211)
    fft = np.fft.fftshift(np.fft.fft(input_signal*window_func(len(input_signal)),FFT_points))
    if normalise:
        freq=20*np.log10(np.abs(fft)/np.max(np.max(np.abs(fft))))
        ax.set_ylim(magnitude_bound,10)

    else:
        freq=20*np.log10(np.abs(fft))
        ax.set_ylim(magnitude_bound,120)


    if not bpm:
        freqs = np.fft.fftshift(np.fft.fftfreq(n=FFT_points, d=sample_period))
        ax.plot(freqs,freq, label = "Frequency Content")

    else:
        freqs = np.fft.fftshift(np.fft.fftfreq(n=FFT_points, d=sample_period))*60
        ax.plot(freqs,freq, label = "Frequency Content")

    if expected_rate:
        ax.plot(np.ones(2)*expected_rate,[np.min(freq),np.max(freq)], label = "Expected Rate")

    
    ax.set_xlim(0,frequency_bound)
    
    if not bpm:
        ax.set_xlabel("Frequency [Hz]")
    else:
        ax.set_xlabel("Frequency [Bpm]")
    ax.set_ylabel("Normalised Magnitude [dB]")
    ax.set_title("%s Frequency Content" % source)

    ax1 = fig.add_subplot(212)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Displacement [m]")
    # ax1.set_title("%s Time Selection" % source)
    ax1.plot(time_vals,input_signal)

    ax.legend()

from numpy.typing import NDArray

def sinc_interpolation(x: NDArray, s: NDArray, u: NDArray) -> NDArray:
            """Whittaker–Shannon or sinc or bandlimited interpolation.
            Args:
                x (NDArray): signal to be interpolated, can be 1D or 2D
                s (NDArray): time points of x (*s* for *samples*) 
                u (NDArray): time points of y (*u* for *upsampled*)
            Returns:
                NDArray: interpolated signal at time points *u*
            Reference:
                This code is based on https://gist.github.com/endolith/1297227
                and the comments therein.
            TODO:
                * implement FFT based interpolation for speed up
            """
            sinc_ = np.sinc((u - s[:, None])/(s[1]-s[0]))

            return np.dot(x, sinc_)

def resample_and_sync(time_1, series_1, time_2):
    """Resample series_1 so that it has time_2. 
    
    Uses linear interpolation. time_2 needs to have a faster sampling rate than time_1.
    The domain of time_1 needs to be a subset of time_2.
    """
    

    up_sampled_time_series = np.zeros(shape=time_2.shape)
    for i in range(len(time_2)):
        # get indices of time_1 before and after time_2
        time_2i = time_2[i]
        time_1i_before_2i = np.argwhere(time_1<time_2i)[-1][0]
        time_1i_after_2i = np.argwhere(time_1>time_2i)[0][0]

        # get the two points at those indices from time1 and series1  
        time_1_before_2i = time_1[time_1i_before_2i]
        time_1_after_2i = time_1[time_1i_after_2i]
        series_1_val_before_2i = series_1[time_1i_before_2i]
        series_1_val_after_2i = series_1[time_1i_after_2i]

        # calculate the value of a point at time_2i on the line
        new_val = np.interp(time_2i, [time_1_before_2i,time_1_after_2i], [series_1_val_before_2i,series_1_val_after_2i])
        up_sampled_time_series[i] = new_val # gradient*time_2i+c

        # print progress bar
        num_increments = 50
        progress = int((i-1)*num_increments/(len(time_2)))
        bar = "".join([u"\u2588"]*progress + [" "]*(num_increments-progress-1))
    #     print("Progress: %d%%" % ((progress+1)*100/num_increments) + " |" + str(bar) + "|"  ,end="\r")

    # print()
    # print("Done.\n")
        
    return up_sampled_time_series


def remove_impulses(time_series,max_change=0.2,number_of_runs=20):
    for i in range(number_of_runs):
        impulse_indices_x = (np.transpose(np.argwhere(np.abs(np.diff(time_series))>=max_change))+1)[0]

        for index in impulse_indices_x:
            time_series[index] = (time_series[index-5] + time_series[index+5])/2

    return time_series

def remove_discontinuities(time_vals, time_series, jump_height_range = (0.0001,0.0003), min_time_between_jumps=0.02):
    
    # find possible discontinuties in signal
    diff = np.diff(time_series)
    bool_arr = np.logical_and(np.abs(diff)>jump_height_range[0] , np.abs(diff)<jump_height_range[1])
    indices = np.argwhere(bool_arr)

    diff_time = time_vals[1:]

    previous_time = 0
    indices_2 = []
    for i in indices:
        j = int(i)

        # check if indices are too close together
        if diff_time[j]-previous_time>=min_time_between_jumps:
            previous_time = diff_time[j]
            indices_2.append(j)

        else:
            continue

        # remove the disontinuity
        time_series[j+1:] = time_series[j+1:] - diff[j]
    
    return time_series
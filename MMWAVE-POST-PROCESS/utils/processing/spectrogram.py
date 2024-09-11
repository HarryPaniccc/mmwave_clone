import numpy as np
from utils.processing.radar_ffts import range_fft
import matplotlib.pyplot as plt
from utils.processing.functions import resample_and_sync, root_mean_square_error,percentage_accuracy
from utils.processing.filters import moving_average_filter


def plot_spectrogram(time,signal,sample_period,window_duration,
                    expected_time=None,expected_frequency=None,pad_ratio=5,overlap_factor=0.99,normalised=True):
    ## FFT SPECTROGRAM BREATHING -------------------------------------------------------------------------------------------------------------------------------------------
    # print("PSD SPECTROGRAM:")

    time_arr, freq_arr, spectro_map = psd_spectrogram(time,signal,
                                                sample_period,window_duration,pad_ratio,overlap_factor,normalised=normalised)
    
    start_frequency = 0
    end_frequency = 4
    freq_start_index = np.argwhere(freq_arr<=start_frequency)[-1][0]
    freq_end_index = np.argwhere(freq_arr<=end_frequency)[-1][0]
    plotted_freq_arr = freq_arr[freq_start_index:freq_end_index]*60


    if start_frequency>=0:
        spectro_map = np.flipud(spectro_map[freq_start_index:freq_end_index,:])
    else:
        spectro_map = spectro_map[freq_start_index:freq_end_index,:]

    # extent_list = [time_arr[0],time_arr[-1],start_frequency,end_frequency]
    # x_ticks = np.linspace(time_arr[0],time_arr[-1],11)

    # freq_inc = 10
    # y_ticks = np.arange(start_frequency*60,end_frequency*60+freq_inc,freq_inc)
    
    # breathing_fig = plt.figure( figsize=(8, 6))
    # img = breathing_fig.add_subplot(111)

    # im = img.imshow(spectro_map,extent=extent_list,cmap="plasma",aspect='auto')
    # img.set_xticks(x_ticks)
    # img.set_yticks(y_ticks/60)
    # breathing_fig.colorbar(im,label="Normalised Signal Strength")
    # img.set_xticklabels(np.round(x_ticks,2))
    # img.set_yticklabels((np.round(y_ticks,2)))
    # img.grid(which='minor', color='w', linestyle='-', linewidth=2)
    # img.set_ylabel("Frequency [bpm]")
    # img.set_xlabel("Window Start Time [s]")
    # img.set_title("Heart Rate Spectrogram [%.2fs Window]" % (window_duration))

    # img.set_ylim(0,4)
    
    # plot expected heart rate onto spectrogram
    measured_rate = []
    time_index = 0
    for t in time_arr:
        slice_of_spectro = spectro_map[:,time_index]
        measured_rate_index = np.argmax(slice_of_spectro)
        measured_rate.append(240-plotted_freq_arr[measured_rate_index])
        time_index += 1
    

    measured_rate = np.array(measured_rate)
    num_pts = 25
    averaged_rate = moving_average_filter(measured_rate,np.clip(len(measured_rate)//3,3,25))
    
    try:
        expected_heart_rate = resample_and_sync(expected_time,expected_frequency,time_arr)
        # img.plot(time_arr,expected_heart_rate/60,"cyan",label="Expected Heart Rate")

    except:
        print("Failed to Sync")
        pass
    
    # img.plot(time_arr,measured_rate/60,"g",label="Measured Heart Rate")
    # img.plot(time_arr,averaged_rate/60,"r",label="Averaged Heart Rate")
    # img.legend()
    
    # accuracy = percentage_accuracy(averaged_rate,expected_heart_rate)
    # rmse = root_mean_square_error(averaged_rate,expected_heart_rate)

    # print("Mean Accuracy: %.2f%%" % (np.mean(accuracy)))
    # print("RMS Error: %.2f bpm" % (rmse))
    # print("Range Normalised RMS Error: %.2f %%" % (rmse*100/((60-12))))
    # print()
    
    return averaged_rate, expected_heart_rate
    
    



def psd_spectrogram(time,signal,sample_period,window_duration,pad_ratio,
                overlap_factor=0.9,
                window_func=np.hanning,
                normalised=False):
    # left aligned
    # get valid time domain given window

    window_start_time_0 = time[0]
    window_end_time_0 =  window_start_time_0 + window_duration
    start_index = np.argwhere(time<=window_start_time_0)[-1][0]
    end_index = np.argwhere(time<=window_end_time_0)[-1][0]
    window_size = len(time[start_index:end_index])
    overlap = np.floor(overlap_factor*window_size)

    window = window_func(window_size)
    FFT_points = window_size*(1+pad_ratio)

    shift = np.floor(window_size - overlap)
    if shift < 1:
        shift = 1

    out_time = []
    spectrogram_map = None
    num_shifts = range(0,int(len(signal)-window_size), int(shift))[-1]
    out_freqs = np.fft.fftshift(np.fft.fftfreq(n=FFT_points, d=sample_period)) # the y axis of the spectrogram in Hz
    df = np.mean(np.diff(out_freqs))
    for i in range(0,int(len(signal)-window_size), int(shift)):

        # add the time of the start of the current window to the output times
        out_time.append(time[i])

        # window the signal
        filtered_signal = signal[i:i+window_size] # filter_butter(signal[i:i+window_size],"highpass",1,4,sample_period)[0]
        observation_window = np.array(np.multiply(window,filtered_signal))
        
        # compute fft
        window_slice = np.fft.fftshift(np.fft.fft(observation_window, FFT_points))
        window_slice = np.flip(window_slice)
        
        psd = np.abs((window_slice**2)*(FFT_points/window_size)/(2*df))
        if normalised:
            psd = psd/np.max(psd)
            
        else:
            pass
        normalised_window_slice = psd
        if i == 0:
            # if it is first time create the spectrogram map using the fft of the first window with padding
            spectrogram_map = normalised_window_slice
        else:
            # if it is not first time add to the spectrogram map using the fft of the current window with padding
            spectrogram_map = np.vstack((spectrogram_map,normalised_window_slice))

        # print progress bar
        num_increments = 50
        progress = int((i-1)*num_increments/(num_shifts))
        bar = "".join([u"\u2588"]*progress + [" "]*(num_increments-progress-1))
    #     print("Progress: %d%%" % ((progress+1)*100/num_increments) + " |" + str(bar) + "|"  ,end="\r")

    # print()

    out_time = np.array(out_time) # the x-axis of the spectrogram
    spectrogram_map = np.transpose(spectrogram_map)
    # spectrogram_map = 20*np.log10(np.abs(spectrogram_map)/np.max(np.max(np.abs(spectrogram_map))))
    
    return out_time, out_freqs, spectrogram_map

def time_series_spectrogram(time,signal,sample_period,window_duration,pad_ratio,
                overlap_factor=0.9,
                window_func=np.hanning,
                normalised=False):
    # left aligned
    # get valid time domain given window

    window_start_time_0 = time[0]
    window_end_time_0 =  window_start_time_0 + window_duration
    start_index = np.argwhere(time<=window_start_time_0)[-1][0]
    end_index = np.argwhere(time<=window_end_time_0)[-1][0]
    window_size = len(time[start_index:end_index])

    
    overlap = np.floor(overlap_factor*window_size)

    window = window_func(window_size)
    FFT_points = window_size*(1+pad_ratio)

    shift = np.floor(window_size - overlap)
    if shift < 1:
        shift = 1

    out_time = []
    spectrogram_map = None
    num_shifts = range(0,int(len(signal)-window_size), int(shift))[-1]
    for i in range(0,int(len(signal)-window_size), int(shift)):

        # add the time of the start of the current window to the output times
        out_time.append(time[i])

        # window the signal
        filtered_signal = signal[i:i+window_size] # filter_butter(signal[i:i+window_size],"highpass",1,4,sample_period)[0]
        observation_window = np.array(np.multiply(window,filtered_signal))
        
        # compute fft
        window_slice = np.fft.fftshift(np.fft.fft(observation_window, FFT_points))
        if normalised:
            normalised_window_slice = 20*np.log10(np.abs(window_slice)/np.max(np.max(np.abs(window_slice))))
        else:
            normalised_window_slice = 20*np.log10(np.abs(window_slice)/1) # reference_amplitude

        if i == 0:
            # if it is first time create the spectrogram map using the fft of the first window with padding
            spectrogram_map = normalised_window_slice
        else:
            # if it is not first time add to the spectrogram map using the fft of the current window with padding
            spectrogram_map = np.vstack((spectrogram_map,normalised_window_slice))

        # print progress bar
        num_increments = 50
        progress = int((i-1)*num_increments/(num_shifts))
        bar = "".join([u"\u2588"]*progress + [" "]*(num_increments-progress-1))
        print("Progress: %d%%" % ((progress+1)*100/num_increments) + " |" + str(bar) + "|"  ,end="\r")

    print()

    out_time = np.array(out_time) # the x-axis of the spectrogram
    out_freqs = np.fft.fftshift(np.fft.fftfreq(n=FFT_points, d=sample_period)) # the y axis of the spectrogram in Hz
    spectrogram_map = np.transpose(spectrogram_map)
    # spectrogram_map = 20*np.log10(np.abs(spectrogram_map)/np.max(np.max(np.abs(spectrogram_map))))
    
    return out_time, out_freqs, spectrogram_map


def ard_spectrogram(radar_hdf5,range_time_profile,range_pad,window_function=np.hanning,range_bin_search = (3,3),show_progress = False):
    """NEEDS WORK"""
    
    # get frame numbers
    start_frame, end_frame = radar_hdf5.get_frame_numbers()
    number_of_frames = end_frame - start_frame

    # get radar parameters
    time_between_chirps = radar_hdf5.time_chirp*3
    wavelength = 3e8/radar_hdf5.frequency_centre
    sensitivity = wavelength/(4*np.pi)
    nChirps = radar_hdf5.nChirps
    nVChannels = radar_hdf5.nVChannels

    frame_array = np.zeros((nChirps*number_of_frames,nVChannels),dtype=np.complex128)

    # load frames
    # print("CONSTRUCTING PHASE: ")
    channel_select = np.arange(nVChannels)
    time_vals = []
    previous_timestamp = 0
    for frame_number,i in zip(range(start_frame,end_frame),np.arange(number_of_frames)):
        # get radar frame
        frame_data, timestamp = radar_hdf5.get_frame(frame_number)
        if timestamp < previous_timestamp:
            timestamp = previous_timestamp + np.mean(np.diff(np.array(time_vals)))*nChirps
        previous_timestamp = timestamp
        frame = radar_hdf5.sort_data(frame_data)

        # get radar range profile
        frame = range_fft(frame,range_pad,window_function)

        # select range bin
        range_bin = int(range_time_profile.get_range_bin(timestamp)) 
        frame_array[i*nChirps:(i+1)*nChirps,:] = np.sum(frame[range_bin-range_bin_search[0]:range_bin+range_bin_search[1]+1,:,:],0)

        for chirp in range(nChirps):
            time_vals.append(timestamp+chirp*time_between_chirps)

        # print progress bar
        if show_progress:
            progress = int((frame_number)*20/(number_of_frames+start_frame))
            bar = "".join([u"\u2588"]*progress + [" "]*(20-progress-1))
            print("Progress: %d%%" % ((progress+1)*100/20) + " |" + str(bar) + "| "  ,end="\r") 
    
    if show_progress:
        print()
        print("Done.")


    chirp_array = np.sum(frame_array[:,:],1)



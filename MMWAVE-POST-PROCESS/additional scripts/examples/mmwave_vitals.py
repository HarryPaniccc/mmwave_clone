## utils
from utils.npy.depth import depthNPY
from utils.hdf5.radar import radarHDF5
from utils.hdf5.test_rig import rigHDF5
from utils.hdf5.polar import polarHDF5
from utils.npy.polar import polarNPY

from utils.processing.ard import range_fft

from utils.processing.functions import *

## common python imports
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import json
import glob
from scipy import signal

def load_radar_data(path_to_data):
    print("LOADING RADAR DATA:")
    working_dir = os.path.dirname(__file__)
    radar_hdf5 = radarHDF5(os.path.join(working_dir,path_to_data))
    print("Done.\n")
    return radar_hdf5

def load_process_config(path_to_config):
    working_dir = os.path.dirname(__file__)
    f = open(os.path.join(working_dir,path_to_config))
    pipeline_config = json.load(f)
    return pipeline_config


def main():
    ## Load Config
    pipeline_config = load_process_config('pipeline_config.json')
    range_pad = pipeline_config["RANGE_CONFIG"]["RANGE_PAD"]
    file_number = pipeline_config["POLAR_DATA_NUM"]


    save = pipeline_config["PHASE_CONFIG"]["SAVE_DATA"]
    metres_per_radian = pipeline_config["PHASE_CONFIG"]["RAD_PER_METRE"]
    expNum = pipeline_config["EXPERIMENT_NUMBER"]
    date = pipeline_config["DATE"]

    ## Load Radar data
    path_to_data = "data/{1}_Exp{0}/Radar_Data_Exp{0}.hdf5".format(expNum,date)
    radar_hdf5 = load_radar_data(path_to_data)
    start_frame_num, end_frame_num = radar_hdf5.get_frame_numbers()
    start_frame = start_frame_num + pipeline_config["PHASE_CONFIG"]["START_FRAME_OFFSET"]
    end_frame = end_frame_num - pipeline_config["PHASE_CONFIG"]["END_FRAME_OFFSET"]
    number_of_frames = end_frame - start_frame_num
    sample_period = radar_hdf5.time_chirp*(radar_hdf5.nVChannels//4)
    wavelength = 3e8/radar_hdf5.frequency_centre
    metres_per_radian = wavelength/(4*np.pi)
    nVChannels = radar_hdf5.nVChannels
    nChirps = radar_hdf5.nChirps
    nSamples = radar_hdf5.nSamples

    ## File Contents
    print("FILE CONTENTS:")
    print("%d Frames" % number_of_frames)
    print("Starting Frame Number: %d" % start_frame_num)
    print("End Frame Number: %d \n" % end_frame)
    print(radar_hdf5.file["Comments"]["experiment_setup"][0][0].decode())
    print()

    ## Range Bin Selection
    print("LOAD RANGE-TIME: ")


    # dogs
    # range_bin = 285 # experiment 1, 20 June Nice
    # range_bin = 287 # experiment 2, 20 June Nice
    # range_bin = 280 # experiment 3, 20 June Nice

    # range_bin = 305 # experiment 7, 20 June
    # range_bin = 290 # experiment 8, 20 June
    # range_bin = 270 # experiment 9, 20 June
    
    # range_bin = 270 # experiment 10, 20 June
    # range_bin = 295 # experiment 11, 20 June
    # range_bin = 270 # experiment 12, 20 June Nice

    # # human
    # range_bin = 414 # experiment 1, 2 Aug
    # range_bin = 387 # experiment 1, 8 Aug
    # range_bin = 400 # experiment 2, 8 Aug
    # range_bin = 420 # experiment 4, 8 Aug
    # range_bin = 386 # experiment 3, 28 July
    # range_bin = 390 # experiment 4, 28 July
    # range_bin = 375 # experiment 1, 30 July
    # range_bin = 372 # experiment 2, 30 July

    range_bin = 420
    known_breathing_rate = 30

    print("STACKING FRAMES: ")
    channel_select = np.arange(nVChannels)
    frame_stack = np.zeros(nVChannels)
    time_vals = []
    window_function = signal.windows.hann
    previous_timestamp = 0
    frame_data, unix_start_time = radar_hdf5.get_frame(0)
    frame = radar_hdf5.sort_data(frame_data)
    for frame_number in range(start_frame,end_frame):
        # get radar frame
        frame_data, timestamp = radar_hdf5.get_frame(frame_number)
        if timestamp < previous_timestamp:
            timestamp = previous_timestamp + np.mean(np.diff(np.array(time_vals)))*nChirps
        previous_timestamp = timestamp
        frame = radar_hdf5.sort_data(frame_data)

        frame = range_fft(frame,nSamples,nChirps,nVChannels,range_pad,window_function)

        # range_bin = int(depth.get_range_bin(timestamp)) 
        frame_stack = np.vstack((frame_stack,frame[range_bin,:,:]))

        for chirp in range(nChirps):
            time_vals.append(timestamp+chirp*sample_period)

        # print progress bar
        progress = int((frame_number)*20/(number_of_frames+start_frame))
        bar = "".join([u"\u2588"]*progress + [" "]*(20-progress-1))
        print("Progress: %d%%" % ((progress+1)*100/20) + " |" + str(bar) + "| "  ,end="\r") 
    
    print()
    print("Done.\n")

    print("CALIBRATING PHASE:")
    frame_stack = frame_stack[1:]
    radar_time = np.array(time_vals)
    magnitude_threshold = 800
    for channel in channel_select:
        frame_stack[:,channel] = frame_stack[:,channel] - np.mean(frame_stack[:,channel])
        print("Channel: ",channel, 
            "\t Min:  %.2f" % np.round(np.min(np.abs(frame_stack[:,channel])),2),
            "|\t Mean: %.2f" % np.round(np.mean(np.abs(frame_stack[:,channel])),2),
            "|\t Max:  %.2f" % np.round(np.max(np.abs(frame_stack[:,channel])),2))

    magnitude_threshold = int(input("Mean Magnitude Threshold: "))
    channels = []
    for channel in channel_select:
        if np.mean(np.abs(frame_stack[:,channel])) >= magnitude_threshold:
            channels.append(channel)
    print("Done.\n")
    

    print("CONSTRUCTING PHASE: ")
    channel_select = np.array(channels)
    chirp_array = np.sum(frame_stack[:,channel_select],1)
    phase = np.angle(chirp_array)
    displacement = np.unwrap(phase)*-metres_per_radian
    print("Done.\n")

        
    ## Setup plot
    IQ_fig = plt.figure()
    scat = IQ_fig.add_subplot(121)
    scat.scatter(np.real(chirp_array),np.imag(chirp_array),s=0.1)
    scat.set_title("IQ Data")
    scat.set_ylabel("Imaginary [Q]")
    scat.set_xlabel("Real [I]")

    hist = IQ_fig.add_subplot(122)
    hist.set_title("Magnitude Histogram")
    hist.set_ylabel("Number of Samples")
    hist.set_xlabel("Sample Magnitudes")
    hist.hist(np.abs(chirp_array),100)

    fig = plt.figure( figsize=(8, 6))
    ax1 = fig.add_subplot()
    ax1.set_title("Phase")
    ax1.set_ylabel("Distance Change [m]")
    ax1.set_xlabel("Time [s]")

    radar_time-=unix_start_time
    ax1.plot(radar_time,displacement)
    ax1.set_xlim(radar_time[0],radar_time[-1])
    fig.show()

    print("TRUNCATE PHASE:")
    print("Choose truncation bounds ->")
    start_time_chosen = int(input("Start time: "))
    end_time_chosen = int(input("End time: "))
    if start_time_chosen >= end_time_chosen:
        print("Invalid selection. Exiting...\n")
        exit()
    radar_start_index =   np.argwhere(radar_time>=start_time_chosen)[0][0]
    radar_end_index =  np.argwhere(radar_time<=end_time_chosen)[-1][0]
    radar_time=radar_time[radar_start_index:radar_end_index]
    displacement=displacement[radar_start_index:radar_end_index]
    print("Done.\n")
    
    ## Save Data for Next Step
    if False:
        print("SAVING PHASE:")

        np.save("radar_phase_time", radar_time)
        np.save("radar_phase" , displacement)
        print("Done.\n")

    
    # filtering
    breathing = filter_butter(displacement,"bandpass",(0.2,20),4,np.mean(np.diff(radar_time)))[0]
    heartrate = filter_butter(displacement,"bandpass",(0.8,3.2),4,np.mean(np.diff(radar_time)))[0]

    fig = plt.figure()
    breathing_plot = plt.subplot(211)
    breathing_plot.plot(radar_time,breathing, label="0.2-0.6Hz")
    breathing_plot.set_ylabel("Displacement [m]")
    breathing_plot.set_title("Bandpass Filtered Phase Signal")
    breathing_plot.legend()


    heart_plot = plt.subplot(212)
    heart_plot.plot(radar_time,heartrate, label="0.8-3.2Hz")
    heart_plot.set_ylabel("Displacement [m]")
    heart_plot.set_xlabel("Time [s]")
    heart_plot.legend()


    


    ## Get Polar Heart rata
    print("LOADING POLAR H10 DATA:")
    polar=None

    working_dir = os.path.dirname(__file__)
    
    try:
        try:
            path_to_data = os.path.join(working_dir,"data/{0}_Exp{1}/Polar_Data_Exp{1}.hdf5".format(date,expNum))
            polar = polarHDF5(path_to_data)
        except:
            path_to_data = "data/{0}_Polar/Polar_Data_{0}_File{1}.npy".format(date,file_number)
            polar = polarNPY(os.path.join(working_dir,path_to_data))
        valid_heart_timestamps = []
        valid_heart_rates = []
        for i in range(len(polar.heart_rate)):
            if (polar.heart_timestamps[i]-unix_start_time)>=radar_time[0] and ((polar.heart_timestamps[i]-unix_start_time)<=radar_time[-1]):
                # print("Heart rate at t=%.2f is %.2fbpm" % (polar.timestamps[i]-unix_start_time,polar.heart_rate[i]))
                valid_heart_timestamps.append(polar.heart_timestamps[i]-unix_start_time)
                valid_heart_rates.append(polar.heart_rate[i])
    except:
        valid_heart_rates = np.zeros(1000)
        valid_heart_timestamps = np.arange(1000)
        print("No Heart rate Recorded")
    
    # if valid_heart_rates == []:
    #     valid_heart_rates = 80*np.ones(len(heartrate))
    #     valid_heart_timestamps = radar_time
    # print("Done.\n")
    
    print("ZERO CROSSINGS:")
    num_pts = 15
    zero_crossings = np.argwhere(np.abs(heartrate)<=0.000002)
    previous_time = radar_time[zero_crossings[0]]
    min_duration = (1/4)/2
    count = 0
    times = []

    for index in zero_crossings:
        if radar_time[index]-previous_time>=min_duration:
            previous_time = radar_time[index]
            times.append(radar_time[index])
            count+=1
            plt.scatter(radar_time[index],heartrate[index],c='r')

    times = np.transpose(np.array(times))[0]
    heart_rates = 60/moving_average_filter(np.diff(times)*2,20)
    heart_rates = moving_average_filter(heart_rates,num_pts)
    accuracy = percentage_accuracy(np.mean(heart_rates),np.mean(valid_heart_rates))

    plt.figure()
    plt.xlabel("Time [s]")
    plt.ylabel("Heart-rate [bpm]]")
    plt.title("Zero Crossing Heart-rate vs Time")
    plt.plot(times[1:],heart_rates,label="Zero Crossing")
    plt.plot(valid_heart_timestamps,valid_heart_rates,label="Polar")
    plt.legend()
    plt.ylim(0,200)


    print("Calculated Mean Heart-rate: %.2f" % np.mean(heart_rates))
    print("Expected Mean Heart-rate: %.2f" % np.mean(valid_heart_rates))
    print("Mean Accuracy: %.2f" % np.mean(accuracy))
    print("Done.\n")

    window_duration = 9 
    print("BREATHING SPECTROGRAM:")


    time_arr, freq_arr, spectro_map = spectrogram(radar_time,breathing,
                                                  sample_period,window_duration,10,0.99,window_func=window_function)
    
    start_frequency = 0
    end_frequency = 4
    freq_start_index = np.argwhere(freq_arr<=start_frequency)[-1][0]
    freq_end_index = np.argwhere(freq_arr<=end_frequency)[-1][0]
    plotted_freq_arr = freq_arr[freq_start_index:freq_end_index]*60


    if start_frequency>=0:
        spectro_map = np.flipud(spectro_map[freq_start_index:freq_end_index,:])
    else:
        spectro_map = spectro_map[freq_start_index:freq_end_index,:]

    extent_list = [time_arr[0],time_arr[-1],start_frequency,end_frequency]
    x_ticks = np.linspace(time_arr[0],time_arr[-1],11)

    freq_inc = 10
    y_ticks = np.arange(start_frequency*60,end_frequency*60+freq_inc,freq_inc)
    

    breathing_fig = plt.figure( figsize=(8, 6))
    img = breathing_fig.add_subplot(111)

    im = img.imshow(spectro_map,extent=extent_list,cmap="plasma",aspect='auto')
    img.set_xticks(x_ticks)
    img.set_yticks(y_ticks/60)
    breathing_fig.colorbar(im,label="Normalised Frequency Component Strength [dB]")
    img.set_xticklabels(np.round(x_ticks,2))
    img.set_yticklabels((np.round(y_ticks,2)))
    img.grid(which='minor', color='w', linestyle='-', linewidth=2)
    img.set_ylabel("Frequency [bpm]")
    img.set_xlabel("Window Start Time [s]")
    img.set_title("Breathing Phase Data Spectrogram [%.2fs Window]" % (window_duration))

    img.set_ylim(0,4)
    
    # plot expected heart rate onto spectrogram
    measured_rate = []
    expected_breathing_rate = []
    time_index = 0
    for t in time_arr:
        expected_breathing_rate.append(known_breathing_rate)
        slice_of_spectro = spectro_map[:,time_index]
        measured_rate_index = np.argmax(slice_of_spectro)
        measured_rate.append(240-plotted_freq_arr[measured_rate_index])
        time_index += 1
    
    expected_breathing_rate = np.array(expected_breathing_rate)
    measured_rate = np.array(measured_rate)
    breathing_rates = moving_average_filter(measured_rate,num_pts)

    img.plot(time_arr,expected_breathing_rate/60,label="Expected Breathing Rate")
    img.plot(time_arr,measured_rate/60,label="Measured Breathing Rate")
    img.plot(time_arr,breathing_rates/60,label="Averaged Breathing Rate")
    img.legend()

    accuracy = percentage_accuracy(breathing_rates,expected_breathing_rate)
    rmse = root_mean_square_error(breathing_rates,expected_breathing_rate)
    error = np.abs(breathing_rates-expected_breathing_rate)

    print("Max Error: %dbpm, Min Error: %dbpm, Mean Error: %dbpm" % (np.max(error),np.min(error),np.mean(error)))
    print("RMS Error: %.2fbpm" % (rmse))

    print("Min Accuracy: %.2f%%" % (np.min(accuracy)))
    print("Mean Accuracy: %.2f%%" % (np.mean(accuracy)))
    print("Max Accuracy: %.2f%%" % (np.max(accuracy)))

    print("Done.\n")

    ## FFT SPECTROGRAM BREATHING -------------------------------------------------------------------------------------------------------------------------------------------
    print("FFT HEART RATE SPECTROGRAM:")
    # time_arr, freq_arr, spectro_map = spectrogram(radar_time,heart_rate,
    #                                               sample_period,window_duration,10,0.99,window_func=window_function)
    time_arr, freq_arr, spectro_map = spectrogram(radar_time,heartrate,
                                                  sample_period,window_duration,10,0.99,window_func=window_function)
    start_frequency = 0
    end_frequency = 4
    freq_start_index = np.argwhere(freq_arr<=start_frequency)[-1][0]
    freq_end_index = np.argwhere(freq_arr<=end_frequency)[-1][0]
    plotted_freq_arr = freq_arr[freq_start_index:freq_end_index]*60
    # print(plotted_freq_arr)
    # exit()

    if start_frequency>=0:
        spectro_map = np.flipud(spectro_map[freq_start_index:freq_end_index,:])
    else:
        spectro_map = spectro_map[freq_start_index:freq_end_index,:]

    extent_list = [time_arr[0],time_arr[-1],start_frequency,end_frequency]
    x_ticks = np.linspace(time_arr[0],time_arr[-1],11)

    freq_inc = 10
    y_ticks = np.arange(start_frequency*60,end_frequency*60+freq_inc,freq_inc)
    

    fig3 = plt.figure( figsize=(8, 6))
    img = fig3.add_subplot(111)

    im = img.imshow(spectro_map,extent=extent_list,cmap="plasma",aspect='auto')
    img.set_xticks(x_ticks)
    img.set_yticks(y_ticks/60)
    fig3.colorbar(im,label="Normalised Frequency Component Strength [dB]")
    img.set_xticklabels(np.round(x_ticks,2))
    img.set_yticklabels((np.round(y_ticks,2)))
    img.grid(which='minor', color='w', linestyle='-', linewidth=2)
    img.set_ylabel("Frequency [bpm]")
    img.set_xlabel("Window Start Time [s]")
    img.set_title("Aterial Phase Data Spectrogram [%.2fs Window]" % (window_duration))

    img.set_ylim(0,4)
    
    # plot expected heart rate onto spectrogram
    expected_heart_rate_spect = []
    measured_rate = []
    time_index = 0
    for t in time_arr:
        try:
            heart_index = np.argwhere(valid_heart_timestamps<=t)[-1][0]
            expected_heart_rate_spect.append(valid_heart_rates[heart_index])
        except:
            expected_heart_rate_spect.append(valid_heart_rates[0])
        slice_of_spectro = spectro_map[:,time_index]
        measured_rate_index = np.argmax(slice_of_spectro)
        measured_rate.append(240-plotted_freq_arr[measured_rate_index])
        time_index += 1
    
    expected_heart_rate_spect = np.array(expected_heart_rate_spect)
    measured_rate = np.array(np.array(measured_rate))
    moving_rate = moving_average_filter(np.array(measured_rate),num_pts)

    
    accuracy = percentage_accuracy(moving_rate,expected_heart_rate_spect)
    rmse = root_mean_square_error(moving_rate,expected_heart_rate_spect)
    error = np.abs(moving_rate-expected_heart_rate_spect)

    print("Max Error: %dbpm, Min Error: %dbpm, Mean Error: %dbpm" % (np.max(error),np.min(error),np.mean(error)))
    print("RMS Error: %.2fbpm" % (rmse))

    print("Min Accuracy: %.2f%%" % (np.min(accuracy)))
    print("Mean Accuracy: %.2f%%" % (np.mean(accuracy)))
    print("Max Accuracy: %.2f%%" % (np.max(accuracy)))

    print("Done.\n")

    img.plot(time_arr,expected_heart_rate_spect/60,label="Expected Heart Rate")
    img.plot(time_arr,measured_rate/60,label="Measured Heart Rate")
    img.plot(time_arr,moving_rate/60,label="Average Heart Rate")
    img.legend()  

    fig3.show()


    plt.show()


    
if __name__ == "__main__":
    main()
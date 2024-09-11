## utils
from utils.hdf5.radar import radarHDF5
from utils.hdf5.polar import polarHDF5
from utils.processing.filters import *
from utils.processing.functions import *
from utils.processing.spectrogram import *




## common python imports
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import time
from scipy import signal
import cv2



def main():
        
    # setup plots
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['font.size'] = 13

    ## load config
    print("LOADING CONFIG DATA:")
    working_dir = os.path.dirname(__file__)
    f = open(os.path.join(working_dir,'pipeline_config.json'))
    pipeline_config = json.load(f)
    data_dir = pipeline_config["FILE_PATH"]

    print("Done.\n")

    ## Open radar data file
    print("LOADING RADAR DATA:")
    radar_hdf5 = radarHDF5(os.path.join(working_dir,data_dir,"Radar_Data_Exp1.hdf5"))
    _, unix_start_time = radar_hdf5.get_frame(0)
    _, end_frame = radar_hdf5.get_frame_numbers()
    _, unix_end_time = radar_hdf5.get_frame(end_frame)

    print("Done.\n")
    
    ## Open polar data file
    print("LOADING POLAR DATA:")
    path_to_data = os.path.join(working_dir,data_dir,"Polar_Data_Exp1.hdf5")
    polar = polarHDF5(path_to_data,accelerometer_data=True)
    print("Done.\n")


    polar_acc_x = polar.accel_x
    polar_acc_y = polar.accel_y
    polar_acc_z = polar.accel_z
    polar_time = np.linspace(0,unix_end_time-unix_start_time,len(polar_acc_x))
    sample_period= np.mean(np.diff(polar_time))

    # plot acceleration
    plt.figure()
    plt.subplot(411)
    plt.title("Acceleration [m/s\u00b2]")
    plt.plot(polar_time,polar_acc_x)
    plt.ylabel("X")

    plt.subplot(412)
    plt.plot(polar_time,polar_acc_y)
    plt.ylabel("Y")

    plt.subplot(413)
    plt.plot(polar_time,polar_acc_z)
    plt.ylabel("Z")

    plt.subplot(414)
    polar_acc = moving_average_filter(polar_acc_z+polar_acc_y+polar_acc_x,20)
    polar_acc = filter_butter(polar_acc,"bandpass",(1,2.3),4,sample_period)[0]
    plt.plot(polar_time,polar_acc)
    plt.ylabel("Sum")
    plt.xlabel("Time [s]")

    ## FFT SPECTROGRAM BREATHING -------------------------------------------------------------------------------------------------------------------------------------------
    print("FFT HEART SPECTROGRAM:")
    window_duration = 9
    num_pts = 15
    known_breathing_rate = 30

    time_arr, freq_arr, spectro_map = psd_spectrogram(polar_time,polar_acc,
                                               sample_period,window_duration,5,0.99,window_func=np.hanning,normalised=True)
    
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
    img.set_title("Heart Rate Spectrogram [%.2fs Window]" % (window_duration))

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
    img.plot(time_arr,expected_breathing_rate/60,alpha=0.2,label="Expected Heart Rate")
    # img.plot(time_arr,measured_rate/60,label="Measured Breathing Rate")
    img.plot(time_arr,breathing_rates/60,alpha=0.2,label="Averaged Heart Rate")
    img.legend()

    

    print("Done.\n")

    plt.show()
    np.save(os.path.join(working_dir,data_dir,"polar_heart_acc"), breathing_rates)
    np.save(os.path.join(working_dir,data_dir,"polar_time"), time_arr+unix_start_time)




if __name__ == "__main__":
    main()
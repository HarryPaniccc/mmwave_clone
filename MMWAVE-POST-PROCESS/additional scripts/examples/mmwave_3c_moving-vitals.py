## utils
from utils.npy.depth import depthNPY
from utils.hdf5.polar import polarHDF5
from utils.npy.polar import polarNPY
from utils.hdf5.radar import radarHDF5

from utils.processing.functions import *
from utils.processing.filters import *


## common python imports
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import json
from PyEMD import CEEMDAN

def main():

    # setup plots
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['font.size'] = 13
    
    ## Load Config
    print("LOADING CONFIG DATA:")
    working_dir = os.path.dirname(__file__)
    f = open(os.path.join(working_dir,'pipeline_config.json'))
    pipeline_config = json.load(f)

    exp_num = pipeline_config["EXPERIMENT_NUMBER"]
    date = pipeline_config["DATE"]

    # additional configs
    hr_lb = 1 # Hz
    hr_up = 3.2 # Hz
    br_lb = 0.2 # Hz
    br_ub = 1 # Hz
    window_function = signal.windows.hann

    




    print("Done.\n")


    ## Load Radar Info
    print("LOADING RADAR INFO:")
    path_to_data = "data/{1}_Exp{0}/Radar_Data_Exp{0}.hdf5".format(exp_num,date)
    working_dir = os.path.dirname(__file__)
    radar_hdf5 = radarHDF5(os.path.join(working_dir,path_to_data))
    radar_hdf5.display_radar_performance()
    _, unix_start_time = radar_hdf5.get_frame(0)
    print(radar_hdf5.file["Comments"]["experiment_setup"][0][0].decode())
    print("Done.\n")

    ## Load Phase Data
    print("LOADING PHASE DATA:")

    path_to_data = "data/{0}_Exp{1}/depth_time.npy".format(date,exp_num)
    phase_time = np.load(os.path.join(working_dir,path_to_data)) # - unix_start_time
    sample_period  = np.mean(np.diff(phase_time))
    
    path_to_data = os.path.join(working_dir,"data/{0}_Exp{1}/depth_phase.npy".format(date,exp_num))
    phase_data = np.load(path_to_data)
    print("Done.\n")

    # Load depth Data
    max_change = 0.001
    runs = 100
    print("LOADING CFAR VELOCITY DATA:")
    working_dir = os.path.dirname(__file__)
    path_to_data = os.path.join(working_dir,"data/{1}_Exp{0}/depth_cv.npy".format(exp_num,date))
    depth_cv = np.load(path_to_data)
    # depth_cv = remove_impulses(depth_cv,max_change,runs)

    print("Done.\n")

    # Load depth Data
    print("LOADING CFAR DEPTH DATA:")
    working_dir = os.path.dirname(__file__)
    path_to_data = os.path.join(working_dir,"data/{1}_Exp{0}/depth_cd.npy".format(exp_num,date))
    depth_cd = np.load(path_to_data)
    # depth_cd = remove_impulses(depth_cd,max_change,runs)
    print("Done.\n")

    ## Load depth Data
    print("LOADING REALSENSE DEPTH DATA:")
    working_dir = os.path.dirname(__file__)
    path_to_data = os.path.join(working_dir,"data/{1}_Exp{0}/depth_rs.npy".format(exp_num,date))
    depth_rs = np.load(path_to_data)
    # depth_rs = remove_impulses(depth_rs,max_change,runs)

    print("Done.\n")

    #truncate experiment#
    print("SETTING EXPERIMENT TIME DOMAIN:")
    segment = 2
    start_time = phase_time[0] + (segment-1)*30
    end_time = start_time+30

    start_index =   np.argwhere(phase_time>=start_time)[0][0]
    end_index =  np.argwhere(phase_time<=end_time)[-1][0]
    phase_time = phase_time[start_index:end_index]
    phase_data = phase_data[start_index:end_index]
    depth_rs = depth_rs[start_index:end_index]
    depth_cd = depth_cd[start_index:end_index]
    depth_cv = depth_cv[start_index:end_index]

    
    print("Done.\n")


    # match velocity to phase
    scales = np.linspace(0.1,1.2,500)
    rms_errors = []
    for scale in scales:
        rms_errors.append(root_mean_square_error((depth_cv-np.mean(depth_cv))*scale,phase_data-np.mean(phase_data)))
    rms_errors = np.array(rms_errors)
    depth_cv  = depth_cv*scales[np.argmin(rms_errors)]

    # # trying adaptive filter based off amplitude
    # data_to_filter = phase_data-depth_cv
    # lower_bounds = np.linspace(0.8,2.5,50)
    # expected_max_amplitude = 0.003 # m
    # cutoffs = []
    # for bound  in lower_bounds:
    #     potential_heart_beat = filter_butter(data_to_filter,"bandpass",(bound,hr_up),4,sample_period)[0]
    #     max_amplitude = np.max(np.abs(potential_heart_beat))
    #     if max_amplitude <= expected_max_amplitude:
    #         cutoffs.append(bound)

    # cutoffs = np.array(cutoffs)
    # # print(cutoffs)
    # hr_lb = cutoffs[0]
    # print(hr_lb)

    # lower_bounds = np.linspace(0.2,0.5,20)
    # upper_bounds = np.linspace(0.7,1,20)

    # expected_max_amplitude = 0.002 # m
    # cutoffs = []
    # for l_bound, u_bound  in zip(lower_bounds,upper_bounds):
    #     potential_breath = filter_butter(data_to_filter,"bandpass",(l_bound,u_bound),4,sample_period)[0]
    #     potential_breath_magnitude = envelope_detector(phase_time,potential_breath,0.02)
    #     potential_breath_magnitude = potential_breath_magnitude*np.max(potential_breath)/np.max(potential_breath_magnitude)
    #     potential_breath_magnitude = np.mean(potential_breath_magnitude)
    #     print(potential_breath_magnitude)

    #     # max_amplitude = np.max(np.abs(potential_breath))
    #     if potential_breath_magnitude <= expected_max_amplitude:
    #         cutoffs.append((l_bound,u_bound))

    # cutoffs = np.array(cutoffs)
    # cutoffs = np.mean(cutoffs,0)
    # br_lb = cutoffs[0]
    # br_ub = cutoffs[1]

    
    ## Get Polar Heart rata
    print("LOADING POLAR H10 DATA:")
    polar=None

    working_dir = os.path.dirname(__file__)
    
    try:
        path_to_data = os.path.join(working_dir,"data/{0}_Exp{1}/Polar_Data_Exp{1}.hdf5".format(date,exp_num))
        polar = polarHDF5(path_to_data)
        valid_heart_timestamps = []
        valid_heart_rates = []
        for i in range(len(polar.heart_rate)):
            if (polar.heart_timestamps[i]-unix_start_time)>=phase_time[0] and ((polar.heart_timestamps[i]-unix_start_time)<=phase_time[-1]):
                valid_heart_timestamps.append(polar.heart_timestamps[i]-unix_start_time)
                valid_heart_rates.append(polar.heart_rate[i])
    except:
        valid_heart_rates = np.zeros(1000)
        valid_heart_timestamps = np.arange(1000)
        print("No Heart rate Recorded")
        print("Done.\n")
    

    plt.show()

    ## Motion Compensation
    print("FILTERING:")

    filter_fig = plt.figure(figsize=(6,8))
    comparison_plot = filter_fig.add_subplot(311)
    comparison_plot.plot(phase_time,depth_rs, label = "Realsense")
    comparison_plot.plot(phase_time,depth_cd, label = "Radar - R")
    comparison_plot.plot(phase_time,depth_cv, label = "Radar - V")
    comparison_plot.plot(phase_time,phase_data, label = "Radar Phase")
    comparison_plot.set_ylabel("Full Bandwidth")
    comparison_plot.set_title("Displacement Comparison in Different Bandwidths")
    comparison_plot.legend()

    # phase_data = phase_data - gait_interference

    phase_breathing, sos = filter_butter(phase_data,"bandpass",(br_lb,br_ub),4,sample_period)
    depth_rs_breathing, sos = filter_butter(depth_rs,"bandpass",(br_lb,br_ub),4,sample_period)
    depth_cd_breathing, sos = filter_butter(depth_cd,"bandpass",(br_lb,br_ub),4,sample_period)
    depth_cv_breathing, sos = filter_butter(depth_cv,"bandpass",(br_lb,br_ub),4,sample_period)

    #scale velocity 
    for i in range(20):
        scales = np.linspace(0.1,2,100)
        rms_errors = []
        for scale in scales:
            rms_errors.append(root_mean_square_error(depth_cv_breathing*scale,phase_breathing))
        rms_errors = np.array(rms_errors)
        depth_cv_breathing  = depth_cv_breathing*scales[np.argmin(rms_errors)]

    breathing_band_plot = filter_fig.add_subplot(312)
    breathing_band_plot.plot(phase_time,depth_rs_breathing, label = "Realsense")
    breathing_band_plot.plot(phase_time,depth_cd_breathing, label = "Radar - R")
    breathing_band_plot.plot(phase_time,depth_cv_breathing, label = "Radar - V")
    breathing_band_plot.plot(phase_time,phase_breathing, label = "Radar Phase")
    breathing_band_plot.set_ylabel("%.2fHz-%.2fHz [m]" % (br_lb,br_ub))
    breathing_band_plot.legend()

    phase_heart, sos = filter_butter(phase_data,"bandpass",(hr_lb,hr_up),4,sample_period) # (1,20)
    depth_rs_heart, sos = filter_butter(depth_rs,"bandpass",(hr_lb,hr_up),4,sample_period)
    depth_cd_heart, sos = filter_butter(depth_cd,"bandpass",(hr_lb,hr_up),4,sample_period)
    depth_cv_heart, sos = filter_butter(depth_cv,"bandpass",(hr_lb,hr_up),4,sample_period)


    #scale velocity 
    for i in range(20):
        scales = np.linspace(0.1,2,100)
        rms_errors = []
        for scale in scales:
            rms_errors.append(root_mean_square_error(depth_cv_heart*scale,phase_heart))
        rms_errors = np.array(rms_errors)
        depth_cv_heart  = depth_cv_heart*scales[np.argmin(rms_errors)]

    heart_band_plot = filter_fig.add_subplot(313)

    heart_band_plot.plot(phase_time,depth_rs_heart,alpha=0.2, label = "Realsense")
    heart_band_plot.plot(phase_time,depth_cd_heart,alpha=0.2, label = "Radar - R")
    heart_band_plot.plot(phase_time,depth_cv_heart, label = "Radar - V")
    heart_band_plot.plot(phase_time,phase_heart, label = "Radar Phase")

    heart_band_plot.set_ylabel("%.2fHz-%.2fHz [m]" % (hr_lb,hr_up))
    heart_band_plot.set_xlabel("Time [s]")
    heart_band_plot.legend()

    print("Done.\n")

    print("MOTION COMPENSATION BAND:")
    breathing = phase_breathing - depth_cv_breathing
    heart =phase_heart- depth_cv_heart  
    print("Done.\n")


    
    # decimate
    print("DECIMATION:")
    final_time = phase_time[-1]
    breathing = signal.decimate(breathing,8)
    heart = signal.decimate(heart,8)
    phase_time = signal.decimate(phase_time,8)
    phase_time = phase_time*(final_time/phase_time[-1])
    sample_period = np.mean(np.diff(phase_time))
    print("New sampling frequency: %.2fHz" % (1/sample_period))
    print("Done.\n")

    
    compensation_fig = plt.figure(figsize=(6,8))
    plt1 = compensation_fig.add_subplot(211)
    plt1.plot(phase_time,breathing, c='b', label = "0.2Hz-1Hz ")
    plt1.set_title("Compensated Vitals")
    plt1.set_ylabel("Displacement [m]")
    plt1.legend()


    plt2 = compensation_fig.add_subplot(212)
    plt2.plot(phase_time,heart, c="r",label = "1Hz-3.2Hz ")
    plt2.set_xlabel("Time [s]")
    plt2.set_ylabel("Displacement [m]")
    plt2.legend()

    # plt.show()


    window_duration = 10
    num_pts = 15

    ## FFT SPECTROGRAM BREATHING -------------------------------------------------------------------------------------------------------------------------------------------
    print("FFT BREATHING SPECTROGRAM:")
    known_breathing_rate = 15

    time_arr, freq_arr, spectro_map = spectrogram(phase_time,breathing,
                                                sample_period,window_duration,5,0.99,window_func=window_function)
    
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
    img.set_title("Breathing Rate Spectrogram [%.2fs Window]" % (window_duration))

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

    img.plot(time_arr,expected_breathing_rate/60,alpha=0.2,label="Expected Breathing Rate")
    img.plot(time_arr,measured_rate/60,alpha=0.2,label="Measured Breathing Rate")
    img.plot(time_arr,breathing_rates/60,alpha=0.2,label="Averaged Breathing Rate")
    img.legend()

    accuracy = percentage_accuracy(breathing_rates,expected_breathing_rate)
    rmse = root_mean_square_error(breathing_rates,expected_breathing_rate)

    # print("Max Error: %dbpm, Min Error: %dbpm, Mean Error: %dbpm" % (np.max(error),np.min(error),np.mean(error)))
    print()
    print("Window duration: %.2fs" % window_duration)
    print("RMS Error: %.2f bpm" % (rmse))
    print("Range Normalised RMS Error: %.2f %%" % (rmse*100/((br_ub-br_lb)*60)))

    print("Mean Accuracy: %.2f%%" % (np.mean(accuracy)))

    print("Done.\n")

    # plt.show()
    # exit()

    ## FFT SPECTROGRAM HEARTRATE -------------------------------------------------------------------------------------------------------------------------------------------
    print("FFT HEART RATE SPECTROGRAM:")

    

    time_arr, freq_arr, spectro_map = spectrogram(phase_time,heart,
                                                sample_period,window_duration,5,0.99,window_func=window_function)


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
    img.set_title("Heart Rate Spectrogram [%.2fs Window]" % (window_duration))

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

    print("Window duration: %.2fs" % window_duration)
    print("RMS Error: %.2fbpm" % (rmse))
    print("Range Normalised RMS Error: %.2f %%" % (rmse*100/((hr_up-hr_lb)*60)))
    print("Mean Accuracy: %.2f%%" % (np.mean(accuracy)))
    print("Done.\n")

    img.plot(time_arr,expected_heart_rate_spect/60,alpha=0.2,label="Expected Heart Rate")
    img.plot(time_arr,measured_rate/60,alpha=0.2,label="Measured Heart Rate")
    img.plot(time_arr,moving_rate/60,alpha=0.2,label="Averaged Heart Rate")
    
    img.legend()    
    fig3.show()


    plt.show()


    
if __name__ == "__main__":
    main()
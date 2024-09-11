from utils.npy.depth import depthNPY
from utils.hdf5.polar import polarHDF5
from utils.npy.polar import polarNPY
from utils.hdf5.radar import radarHDF5

from utils.processing.functions import *
from utils.processing.spectrogram import plot_spectrogram

from utils.processing.filters import *


## common python imports
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate, hilbert
import cv2 as cv
import json
import glob



def main():

    # setup plots
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['font.size'] = 13


    working_dir = os.path.dirname(__file__)
    # load config 
    f = open(os.path.join(working_dir,'pipeline_config.json'))
    pipeline_config = json.load(f)
    data_dir_path = pipeline_config["FILE_PATH"]
    
    path = os.path.join(working_dir,data_dir_path)
    folder = path.split("\\")[-1]
    if folder.split("_")[-1] == "handler":
        exit() # skip experiments with handlder too close to dog
    

    phase_file =  os.path.join(path,"phase","phase_meas_0.npy")

    phase_time = np.load(phase_file)[0] # - unix_start_time
    unix_start_time = phase_time[0]
    phase_time = phase_time - unix_start_time
    
    phase_data = np.load(phase_file)[1] 


    # divide sample rate by 3**4 
    for j in range(4):
        phase_data = decimate(phase_data,3)
    
    phase_time = np.linspace(phase_time[0],phase_time[-1],len(phase_data))
    time_fig = plt.figure()
    original = time_fig.add_subplot(311)
    original.plot(phase_time,phase_data)
    
    filtered = time_fig.add_subplot(312)
    trad_filter, _ = filter_butter(phase_data,"highpass",0.6,4,np.mean(np.diff(phase_time)))
    filtered.plot(phase_time,trad_filter,"r")
    
    filtered = time_fig.add_subplot(313)
    trad_filter, _ = filter_butter(phase_data,"bandpass",(1,3),4,np.mean(np.diff(phase_time)))
    filtered.plot(phase_time,trad_filter,"r")
    
    
    
    # filter bank
    bandwidth = 0.1 # Hz 
    start_frequency = 0.8
    end_frequency = 3.2
    l_bounds = np.arange(start_frequency,end_frequency,bandwidth)
    u_bounds = l_bounds + bandwidth
    
    # 2D Filter
    i = 0
    filter_image = np.zeros((len(l_bounds),len(phase_data)))
    image_envelope = np.zeros((len(l_bounds),len(phase_data)))
    for lc, uc in zip(l_bounds,u_bounds):
        
        notch_signal, _ = filter_butter(phase_data,"bandpass",(lc,uc),4,np.mean(np.diff(phase_time)))
        filter_image[i,:] = notch_signal
        
        analytic_signal = hilbert(notch_signal)
        image_envelope[i,:] = np.abs(analytic_signal)
        
        i+=1      
    
    extent = [phase_time[0],phase_time[-1],end_frequency,start_frequency]
                
    plt.figure()
    plt.subplot(221)
    plt.imshow(filter_image,aspect="auto",extent = extent)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    
    
    plt.subplot(222)
    plt.imshow(image_envelope,aspect="auto",extent = extent)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    
    
    # threshold
    threshold_map = np.ones(filter_image.shape)
    threshold_map[image_envelope>0.0004] = 0
    threshold_map[image_envelope<0.0002] = 0
    filter_gain = cv.GaussianBlur(threshold_map, (255,3),0)
    filter_out_2D = filter_image*threshold_map
    filter_out = np.sum(filter_out_2D,0)
    
    plt.subplot(223)
    plt.imshow(threshold_map,aspect="auto",extent = extent)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    
    plt.subplot(224)
    plt.imshow(filter_out_2D,aspect="auto",extent = extent)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    
    filtered.plot(phase_time,filter_out,"b")
    
    
    ## Get Polar Heart rata

    try:
        exp_num = (glob.glob(os.path.join(working_dir,data_dir_path,"Polar_Data_Exp*"))[0].split(".npy")[0].split("Exp")[-1])
        polar_data = np.load(os.path.join(working_dir,data_dir_path,"Polar_Data_Exp{0}.npy".format(exp_num)))
        polar_time, polar_heart_rate = polar_data.transpose()
        
    except:
        exp_num =  (glob.glob(os.path.join(working_dir,data_dir_path,"Polar_Data_Exp*"))[0].split(".hdf5")[0].split("Exp")[-1])
        polar_heart_rate = np.load(os.path.join(working_dir,data_dir_path,"polar_heart_acc.npy".format(exp_num)))
        polar_time = np.load(os.path.join(working_dir,data_dir_path, "polar_time.npy".format(exp_num)))                        

    if '' in polar_heart_rate:
        filler = float(input("Filler value: "))
    else:
        filler = 0
        
    polar_time = np.array([float(p_time) for p_time in polar_time])  - unix_start_time
    polar_heart_rate = np.array([float(p_heart) if p_heart != "" else filler for p_heart in polar_heart_rate])

    print(polar_time)
    print(phase_time)
    
    plot_spectrogram(phase_time,filter_out,np.mean(np.diff(phase_time)),10,
                        polar_time,polar_heart_rate)
    
    plot_spectrogram(phase_time,trad_filter,np.mean(np.diff(phase_time)),10,
                        polar_time,polar_heart_rate)
    plt.show()

if __name__ == "__main__":
    main()
## utils
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


    dogs = ["Bullet"] # ,"Guster","Kasey","Sassy",
    working_dir = os.path.dirname(__file__)
    data_dir_path = "data/raw_data/"
    
    exp_count = 0
    obsrv_count = 0
    
    
    # Select files
    for dog in dogs:
        print("Dog:", dog)
        for path in glob.glob(os.path.join(working_dir,data_dir_path,dog,"*Exp*")):
            folder = path.split("\\")[-1]
            if folder.split("_")[-1] == "handler":
                continue # skip experiments with handlder too close to dog
            exp_count+=1
            
            date = "_".join(folder.split("_")[0:2])
            exp_num = int(folder.split("_")[-1].split("p")[-1])
            print("\tDate:", date, "\tExp:",exp_num)
            
            try:
                
                idx=0
                phase_folder = os.path.join(path,"phase")
                for phase_file in glob.glob(os.path.join(phase_folder,"phase_*")):
                    
                    ## load data
                    phase_time = np.load(phase_file)[0] # - unix_start_time
                    unix_start_time = phase_time[0]
                    phase_time = phase_time - unix_start_time
                    phase_data = np.load(phase_file)[1] # - unix_start_time

                    
                    start_time = 0 #P2-26 #P3-40
                    end_time =  120 #P2-32 #P3-46
                    
                    hr_lb = 1 # Hz
                    hr_up = 2.3 # Hz
                    window_size_samples = 10*2500 # approximate for 10s 
                    shift = window_size_samples//2 # 50% overlap
                    
                    start_index =   np.argwhere(phase_time>=start_time)[0][0]
                    end_index =  np.argwhere(phase_time<=end_time)[-1][0]
                    phase_time = phase_time[start_index:end_index]
                    phase_data = phase_data[start_index:end_index]

                    # Sample Time
                    sample_period  = np.mean(np.diff(phase_time))
                    
                    ## Get Polar Heart rata
                    try:
                        polar_data = np.load(os.path.join(working_dir,"data","raw_data",dog,date+"_Exp"+str(exp_num),
                                                      "Polar_Data_Exp{0}.npy".format(exp_num)))
                        polar_time, polar_heart_rate = polar_data.transpose()
                        
                    except:
                        polar_heart_rate = np.load(os.path.join(working_dir,"data","raw_data",dog,date+"_Exp"+str(exp_num),
                                                        "polar_heart_acc.npy".format(exp_num)))
                        polar_time = np.load(os.path.join(working_dir,"data","raw_data",dog,date+"_Exp"+str(exp_num),
                                                        "polar_time.npy".format(exp_num)))                        

                    if '' in polar_heart_rate:
                        filler = float(input("Filler value: "))
                    else:
                        filler = 0
                        
                    polar_time = np.array([float(p_time) for p_time in polar_time])  - unix_start_time
                    polar_heart_rate = np.array([float(p_heart) if p_heart != "" else filler for p_heart in polar_heart_rate])
     

                    # filter for heart beat
                    heart_rate, sos = filter_butter(phase_data,"bandpass",(hr_lb,hr_up),4,sample_period)
                    
                    # reduce data content
                    for j in range(4):
                        phase_data = decimate(phase_data,3)
                        
                    phase_time = np.linspace(phase_time[0],phase_time[-1],len(phase_data))
                    
                    ## Select data -------------------------------------------------------------------------------------------------------------------------------------------
                    end_of_data = False
                    i = 0
                    while not end_of_data:
                        phase_data = heart_rate[i:i+window_size_samples]
                        time_data = phase_time[i:i+window_size_samples]
                        start_time = time_data[0]
                        end_time = time_data[-1]
                        
                        time_data = time_data - time_data[0]
                        start_index =   np.argwhere(polar_time>=start_time)[0][0]
                        end_index =  np.argwhere(polar_time<=end_time)[-1][0]
                        
                        mean_heart_rate = np.mean(polar_heart_rate[start_index:end_index])
                        
                        if len(phase_data) < window_size_samples:
                            end_of_data = True
                            break
                        
                        file_name = "{}_exp{}_idx{}_HR__{:.0f}__filtered_phase_data.csv".format(date,exp_num,idx,mean_heart_rate)
                        path = os.path.join(working_dir,"data","canine_dnn_data/Dogs/{}/".format(dog), file_name)
                        
          
                        data = np.transpose(phase_data)
                        np.savetxt(path, data, fmt = '%.9f', delimiter=',', header = 'displacement')
                                
                        
                        i+=shift
                        idx+=1
                        obsrv_count+=1

    

            except Exception as e:
                print(e)
                
                
            continue
             
   
    print("Extracted %d HR observations from %d experiments" % (obsrv_count,exp_count))
    
    

    # with open(path) as csv_file:
    #     lines = len(csv_file.readlines())
    #     csv_file.seek(0)
    #     headers = csv_file.readline()
    #     lines -= 1
        
    #     displacement = np.zeros(lines)
        
    #     for i in range(lines):
    #         displacement[i] = csv_file.readline()

    # plt.plot(phase_time[0:256],displacement*1000,'r')
    # plt.title("Example Data")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Displacement [mm]")
    # plt.show()




    
if __name__ == "__main__":
    main()
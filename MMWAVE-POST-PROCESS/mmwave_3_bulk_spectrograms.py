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


    dogs = ["Paddy","Guster","Kasey","Sassy","Bullet"] # 
    working_dir = os.path.dirname(__file__)
    data_dir_path = "data/raw_data/"
    
    
    # Select files
    rms_errors = []
    accuracy_list = []
            
    for dog in dogs:
        print("Dog:", dog)
        idx = 0
        for path in glob.glob(os.path.join(working_dir,data_dir_path,dog,"*Exp*")):
            folder = path.split("\\")[-1]
            if folder.split("_")[-1] == "handler":
                continue # skip experiments with handlder too close to dog
            
            date = "_".join(folder.split("_")[0:2])
            exp_num = int(folder.split("_")[-1].split("p")[-1])
            print("\tDate:", date, "\tExp:",exp_num)
            
            
            phase_data = 0
            phase_folder = os.path.join(path,"phase")
            for phase_file in glob.glob(os.path.join(phase_folder,"phase_*")):
                                
                ## load data
                phase_time = np.load(phase_file)[0] # - unix_start_time
                unix_start_time = phase_time[0]
                phase_time = phase_time - unix_start_time
                
                try:
                    phase_data += np.load(phase_file)[1] 
                except:
                    phase_data = np.load(phase_file)[1]
                    
                    
                    
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

            polar_time = np.array([float(p_time) for p_time in polar_time])  - unix_start_time
            polar_heart_rate = np.array([float(p_heart) for p_heart in polar_heart_rate])
            
            # reduce data content
            for j in range(4):
                phase_data = decimate(phase_data,3)
        
            phase_time = np.linspace(phase_time[0],phase_time[-1],len(phase_data))
            sample_period  = np.mean(np.diff(phase_time))
                        
            heart_rate, _ = filter_butter(phase_data,"bandpass",(1,2.3),4,sample_period)            
            
            if polar_time[-1]<= phase_time[-1]:
                polar_time =  np.pad(polar_time, (1, 1), 'constant', constant_values=(phase_time[0]-1,phase_time[-1]+1))
                polar_heart_rate = np.pad(polar_heart_rate,(1,1),"edge")
                
            averaged_rate, expected_heart_rate = plot_spectrogram(phase_time,heart_rate,np.mean(np.diff(phase_time)),10,
                polar_time,polar_heart_rate)
            
            # plt.show()
            accuracy = percentage_accuracy(averaged_rate,expected_heart_rate)
            accuracy_list.append(np.mean(accuracy))
            
            rmse = root_mean_square_error(averaged_rate,expected_heart_rate)
            rms_errors.append(np.mean(rmse))

            np.save(os.path.join(working_dir,"data","heart_data",dog,"Polar_HR_Exp{}_Idx{}".format(exp_num,idx)),expected_heart_rate)
            np.save(os.path.join(working_dir,"data","heart_data",dog,"Radar_HR_Exp{}_Idx{}".format(exp_num,idx)),averaged_rate)
            idx+=1
            
    plt.show()
    accuracy_list = np.array(accuracy_list)
    print("Mean Accuracy: ",np.mean(accuracy_list))
    rms_errors = np.array(rms_errors)
    print("Mean RMSE: ",np.mean(rms_errors))
            
            


    
if __name__ == "__main__":
    
    main()
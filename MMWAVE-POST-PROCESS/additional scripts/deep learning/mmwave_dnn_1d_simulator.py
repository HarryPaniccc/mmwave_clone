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
import json
import glob



def main():

    # setup plots
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['font.size'] = 13

    
    dogs = ["Simulated_Noisy"]
    working_dir = os.path.dirname(__file__)
    
    obsrv_count = 0
    
    # Select files
    for dog in dogs:
            print("Dog:", dog)
        
            
            save = True
            adjustment = 0.1
            hr_lb = 1 - adjustment# Hz
            hr_up = 2.3 + adjustment # Hz
            window_size_samples = 256
            num_observations = 1200
            
            # filter for heart beat
            ti = 0
            tf = 10
            t = np.linspace(ti,tf,window_size_samples)
            
            
            for observation in range(num_observations):
                obsrv_count+=1
                
                mean_heart_rate = np.random.uniform(60,180)
                heart_amplitude = np.random.uniform(0.00015,0.00055)
                heart_frequency = 2*np.pi*(mean_heart_rate/60)
                heart_phase = np.random.uniform(-np.pi/2,np.pi/2)
                noise =  filter_butter(0.0015*np.random.randn(len(t)),"bandpass",(1,2.3),4,np.mean(np.diff(t)))[0]
                amplitude_distortion = 1 + filter_butter(0.2*np.random.randn(len(t)),"bandpass",(0.2,2.3),4,np.mean(np.diff(t)))[0]
                
                phase_data = heart_amplitude*np.sin(heart_frequency*t + heart_phase) +noise
                phase_data = phase_data*amplitude_distortion
                data = np.transpose(phase_data)
                
                # plt.plot(t,phase_data)
                
                # plt.show()
                # exit()
               
                file_name = "noisy_simulated_idx{}_HR__{:.0f}__filtered_phase_data.csv".format(obsrv_count,mean_heart_rate)
                path = os.path.join(working_dir,"canine_dnn_data/Dogs/{}/".format(dog), file_name)
                np.savetxt(path, data, fmt = '%.9f', delimiter=',', header = 'displacement')
                    
                print("{:.2f}%".format(obsrv_count*100/num_observations),end="\r")
                
    print()
    print("Extracted %d HR observations from simulated data" % (obsrv_count))




    
if __name__ == "__main__":
    main()
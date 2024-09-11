## utils
from utils.processing.filters import *

## common python imports
import os
import numpy as np
import matplotlib.pyplot as plt



def main():

    # setup plots
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['font.size'] = 13

    working_dir = os.path.dirname(__file__)
    data_dir_path = "data/canine_dnn_data/Dogs"
    
    # observations grouped in increasing difficulty levels 
    num_observations_lvl1 = 500
    num_observations_lvl2 = 1000
    # num_observations_lvl3 = 1250
    # num_observations_lvl4 = 750    
    
    num_samples = 309
    t = np.linspace(0,10,num_samples)
    
    try:
        os.makedirs(os.path.join(working_dir,data_dir_path,"Simulated_1"))
    except:
        pass
    
    dir = os.path.join(working_dir,data_dir_path,"Simulated_1")
    for i in range(num_observations_lvl1):
        heart_rate = np.random.randint(60,180)
        phase_hr = np.random.uniform(-np.pi,np.pi)
        
        amplitude = 0.0003
        signal = amplitude*np.sin(2*np.pi*heart_rate*t/60 + phase_hr)
        file_name = "idx{}_HR__{:.0f}__simulated_phase_data.csv".format(i,heart_rate)
        file_path = os.path.join(dir,file_name)                       
        data = np.transpose(signal)
        np.savetxt(file_path, data, fmt = '%.9f', delimiter=',', header = 'displacement')
    
    
    try:
        os.makedirs(os.path.join(working_dir,data_dir_path,"Simulated_2"))
    except:
        pass
    
    dir = os.path.join(working_dir,data_dir_path,"Simulated_2")
    for j in range(num_observations_lvl2):
        heart_rate = np.random.randint(60,180)
        phase_hr = np.random.uniform(-np.pi,np.pi)
        amplitude_hr = 0.0003
        heart_sig = amplitude_hr*np.sin(2*np.pi*heart_rate*t/60 + phase_hr)
        
        resp_rate = np.random.randint(15,30)
        phase_br = np.random.uniform(-np.pi,np.pi)
        amplitude_br = 0.0022
        breathing_sig = amplitude_br*np.sin(2*np.pi*resp_rate*t/60 + phase_br)
        noise = np.random.randn(num_samples)*0.00015
        total_sig = heart_sig + breathing_sig +noise
        
        file_name = "idx{}_HR__{:.0f}__simulated_phase_data.csv".format(j,heart_rate)
        file_path = os.path.join(dir,file_name)                       
        data = np.transpose(total_sig)
        np.savetxt(file_path, data, fmt = '%.9f', delimiter=',', header = 'displacement')
    exit()
    
    try:
        os.makedirs(os.path.join(working_dir,data_dir_path,"Simulated_3"))
    except:
        pass
    dir = os.path.join(working_dir,data_dir_path,"Simulated_3")
    for k in range(num_observations_lvl3):
        heart_rate = np.random.randint(60,180)
        phase_hr = np.random.uniform(-np.pi,np.pi)
        amplitude_hr = 0.0003
        heart_sig = amplitude_hr*np.sin(2*np.pi*heart_rate*t/60 + phase_hr)
        
        resp_rate = np.random.randint(15,140) # extends to panting now
        phase_br = np.random.uniform(-np.pi,np.pi)
        amplitude_br = 0.0022
        breathing_sig = amplitude_br*np.sin(2*np.pi*resp_rate*t/60 + phase_br)
        noise = np.random.randn(num_samples)*0.0003 # slightly noisier
        total_sig = heart_sig + breathing_sig +noise
        
        file_name = "idx{}_HR__{:.0f}__simulated_phase_data.csv".format(k,heart_rate)
        file_path = os.path.join(dir,file_name)                       
        data = np.transpose(total_sig)
        np.savetxt(file_path, data, fmt = '%.9f', delimiter=',', header = 'displacement')
    
    try:
        os.makedirs(os.path.join(working_dir,data_dir_path,"Simulated_4"))
    except:
        pass  
    for l in range(num_observations_lvl4):
        pass



    
if __name__ == "__main__":
    main()
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
import json
import glob



def main():

    # setup plots
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['font.size'] = 13

    
    dogs = ["Bullet","Guster","Kasey","Paddy","Sassy"]
    working_dir = os.path.dirname(__file__)
    
    exp_count = 0
    obsrv_count = 0
    
    # Select files
    for dog in dogs:
        print("Dog:", dog)
        for path in glob.glob(os.path.join(working_dir,"data",dog,"*Exp*")):
            exp_count+=1
            folder = path.split("\\")[-1]
            if folder.split("_")[-1] == "handler":
                continue
            date = "_".join(folder.split("_")[0:2])
            exp_num = int(folder.split("_")[-1].split("p")[-1])
            print("\tDate:", date, "\tExp:",exp_num)
            

            ## Load Radar Info
            path_to_data = "data/{2}/{1}_Exp{0}/Radar_Data_Exp{0}.hdf5".format(exp_num,date,dog)
            working_dir = os.path.dirname(__file__)
            radar_hdf5 = radarHDF5(os.path.join(working_dir,path_to_data))

            sample_period  = radar_hdf5.time_chirp*3
            # radar_hdf5.display_radar_performance()


            _, unix_start_time = radar_hdf5.get_frame(0)

            ## Load Phase Data
            working_dir = os.path.dirname(__file__)

            path_to_data = "data/{2}/{0}_Exp{1}/depth_time.npy".format(date,exp_num,dog)
            phase_time = np.load(os.path.join(working_dir,path_to_data)) # - unix_start_time
            
            # Sample Time
            sample_period  = np.mean(np.diff(phase_time))
            
            path_to_data = os.path.join(working_dir,"data/{2}/{0}_Exp{1}/depth_phase.npy".format(date,exp_num,dog))
            phase_data = np.load(path_to_data)

            start_time = 0 #P2-26 #P3-40
            end_time =  80 #P2-32 #P3-46
            
            save = True
            adjustment = 0.1
            hr_lb = 1 - adjustment# Hz
            hr_up = 2.3 + adjustment # Hz
            # window_size_seconds = 7.5
            window_size_samples = 256
            
            
            
            start_index =   np.argwhere(phase_time>=start_time)[0][0]
            end_index =  np.argwhere(phase_time<=end_time)[-1][0]
            phase_time = phase_time[start_index:end_index]
            phase_data = phase_data[start_index:end_index]


            ## Get Polar Heart rata
            polar_error_flag = False
            try:
                working_dir = os.path.dirname(__file__)
                polar_data = np.load(os.path.join(working_dir,"data/{2}/{0}_Exp{1}/Polar_Data_Exp{1}.npy".format(date,exp_num,dog)))
                
                polar_time, polar_heart_rate = polar_data.transpose()

                if '' in polar_heart_rate:
                    filler = float(input("Filler value: "))
                else:
                    filler = 0
                polar_time = np.array([float(p_time) for p_time in polar_time]) - unix_start_time
                polar_heart_rate = np.array([float(p_heart) if p_heart != "" else filler for p_heart in polar_heart_rate])
            except Exception as e:
                polar_error_flag = True


            # Decimate
            for i in range(3):
                # heart_rate = signal.decimate(heart_rate, 6)
                phase_data = signal.decimate(phase_data, 5)
                phase_time = signal.decimate(phase_time, 5)
            sample_period = np.mean(np.diff(phase_time))

   
            
            # filter for heart beat
            heart_rate, sos = filter_butter(phase_data,"bandpass",(hr_lb,hr_up),4,sample_period)
            
            ## FFT SPECTROGRAM HEART -------------------------------------------------------------------------------------------------------------------------------------------
            end_of_data = False
            idx=0
            i = 0
            shift = 100
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
                path = os.path.join(working_dir,"canine_dnn_data/Dogs/{}/".format(dog), file_name)
                
                # data = np.transpose(np.array([time_data,phase_data]))
                # np.savetxt(path, data, fmt = '%.9f', delimiter=',', header = 'time, displacement')
                
                data = np.transpose(phase_data)
                np.savetxt(path, data, fmt = '%.9f', delimiter=',', header = 'displacement')
                        
                
                i+=shift
                idx+=1
                obsrv_count+=1

   
   
    print("Extracted %d HR observations from %d experiments" % (obsrv_count,exp_count))
    
    

    with open(path) as csv_file:
        lines = len(csv_file.readlines())
        csv_file.seek(0)
        headers = csv_file.readline()
        lines -= 1
        
        displacement = np.zeros(lines)
        
        for i in range(lines):
            displacement[i] = csv_file.readline()

    plt.plot(phase_time[0:256],displacement*1000,'r')
    plt.title("Example Data")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.show()




    
if __name__ == "__main__":
    main()
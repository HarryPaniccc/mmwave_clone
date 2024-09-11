## utils
from utils.npy.depth import depthNPY
from utils.hdf5.polar import polarHDF5
from utils.npy.polar import polarNPY
from utils.hdf5.radar import radarHDF5
from utils.hdf5.test_rig import rigHDF5


from utils.processing.functions import *
from utils.processing.filters import *


## common python imports
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import json

def main():

    # setup plots
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['font.size'] = 13

    # print(plt.rcParams.keys())
    # exit()
    ## Load Config
    print("LOADING CONFIG DATA:")
    working_dir = os.path.dirname(__file__)
    f = open(os.path.join(working_dir,'pipeline_config.json'))
    pipeline_config = json.load(f)

    exp_num = pipeline_config["EXPERIMENT_NUMBER"]
    date = pipeline_config["DATE"]


    # additional configs
    window_function = signal.windows.bartlett



    print("Done.\n")


    ## Load Radar Info
    print("LOADING RADAR INFO:")
    path_to_data = "data/{1}_Exp{0}/Radar_Data_Exp{0}.hdf5".format(exp_num,date)
    working_dir = os.path.dirname(__file__)
    radar_hdf5 = radarHDF5(os.path.join(working_dir,path_to_data))

    sample_period  = radar_hdf5.time_chirp*3
    # radar_hdf5.display_radar_performance()
    print(radar_hdf5.file["Comments"]["experiment_setup"][0][0].decode())


    _, unix_start_time = radar_hdf5.get_frame(0)
    print("Done.\n")

    ## Load Phase Data
    print("LOADING DATA:")
    working_dir = os.path.dirname(__file__)

    path_to_data = "data/{0}_Exp{1}/depth_time.npy".format(date,exp_num)
    phase_time = np.load(os.path.join(working_dir,path_to_data)) #- unix_start_time
    
    # Sample Time
    sample_period  = np.mean(np.diff(phase_time))
    
    path_to_data = os.path.join(working_dir,"data/{0}_Exp{1}/depth_phase.npy".format(date,exp_num))
    phase_data = np.load(path_to_data)

    path_to_data = os.path.join(working_dir,"data/{0}_Exp{1}/depth_rs.npy".format(date,exp_num))
    rs_data = np.load(path_to_data)

    path_to_data = os.path.join(working_dir,"data/{0}_Exp{1}/Rig_Data_Exp0.hdf5".format(date,exp_num))
    test_rig = rigHDF5(path_to_data)
    sample_start, sample_end = test_rig.get_sample_numbers()
    sample_numbers = np.arange(sample_start,sample_end+1)
    rig_time = []
    rig_data = []
    for num in sample_numbers:
        try:
            sample_data, timestamp = test_rig.get_sample(num)
        except:
            continue

        rig_time.append(timestamp-unix_start_time)
        rig_data.append(-sample_data)
    
    rig_time = np.array(rig_time)
    rig_data = np.array(rig_data)


    path_to_data = os.path.join(working_dir,"data/{1}_Exp{0}/rs_distance.npy".format(exp_num,date))
    range_time_profile = depthNPY(path_to_data)
    depth_rs = range_time_profile.range_bins # moving_average_filter(range_time_profile.range_bins,10)
    depth_rs_time = range_time_profile.timestamps - unix_start_time
    

    print("Done.\n")

    print("SETTING EXPERIMENT TIME DOMAIN:")
    start_time = 15 #P2-26 #P3-40
    end_time = 35 #P2-32 #P3-46
    start_index =   np.argwhere(phase_time>=start_time)[0][0]
    end_index =  np.argwhere(phase_time<=end_time)[-1][0]
    
    phase_time = phase_time[start_index:end_index]
    
    phase_data = phase_data[start_index:end_index]
    phase_data = phase_data - phase_data[0]

    rs_data = rs_data[start_index:end_index]
    rs_data = rs_data - rs_data[0]


    start_index =   np.argwhere(rig_time>=start_time)[0][0]
    end_index =  np.argwhere(rig_time<=end_time)[-1][0]

    rig_time = rig_time[start_index:end_index]
    rig_data = rig_data[start_index:end_index]
    rig_data = rig_data - rig_data[0]


    start_index =   np.argwhere(depth_rs_time>=start_time)[0][0]
    end_index =  np.argwhere(depth_rs_time<=end_time)[-1][0]

    depth_rs_time = depth_rs_time[start_index:end_index]
    depth_rs = depth_rs[start_index:end_index]
    depth_rs = depth_rs - depth_rs[0]

    # phase_data = filter_butter(phase_data,"lowpass",4,4,sample_period)[0]

    f, (a0, a1) = plt.subplots(2, 1, height_ratios=[3,1])

    macro_motion = filter_ellip(rs_data,"bandstop",(0.07,0.13),4,sample_period,ripple=0.5)[0]
    known_signal = filter_butter(macro_motion,"lowpass",0.3,4,sample_period)[0]
    known_signal = ((np.sin(2*np.pi*(1/60)*(phase_time+20)))**2)/3 + (np.sin(2*np.pi*0.1*(phase_time+20)))/100
    known_signal = known_signal-known_signal[0]

    a0.plot(phase_time,known_signal,label="Pre Known Gross Motion")
    a0.plot(phase_time,rs_data,label="Real-Sense")
    a0.plot(phase_time,phase_data,label="Radar Phase")
    a0.plot(rig_time,rig_data,label="Rig Position Log")
    a0.set_ylabel("Measured Displacement [m]")
    a0.set_title("Test Rig Measurements and Compensation")
    a0.legend()

    compensated_phase  = phase_data - macro_motion
    compensated_phase = compensated_phase - np.mean(compensated_phase)
    a1.plot(rig_time,np.sin(2*np.pi*0.1*(rig_time+1.5))/100,label="Known")
    a1.plot(phase_time,compensated_phase,label="Calculated")
    a1.set_ylabel("Micro Displacement [m]")
    a1.set_xlabel("Time [s]")
    a1.legend()


    plt.figure()
    freqs, freq_mag, _ = get_freqeunecy_content(phase_time,phase_data,100000,False)
    plt.plot(freqs,freq_mag,label="Raw Phase")
    
    freqs, freq_mag, _ = get_freqeunecy_content(phase_time,compensated_phase,100000,False)
    plt.plot(freqs,freq_mag,label="Compensated & Filtered Phase")
    plt.plot([0.1,0.1],[0,-300],'--r',linewidth=1)
    plt.scatter([0.1],[0],s=50,c='r',label="Expected Frequency 0.1Hz",zorder=10)

    plt.xlim(0,4)
    plt.ylim(-150,10)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Normalised Magnitude [dB]")
    plt.title("Frequency Content of Compensated vs Uncompensated Signal")
    plt.legend()

    plt.show()


    
if __name__ == "__main__":
    main()
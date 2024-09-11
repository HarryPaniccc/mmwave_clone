import os
import glob
import numpy as np
import sys
from utils.processing.functions import sort_data

def mmwcas_load_data_from_bin(path_to_data, nSamples,nChirps,nChannels, num_lanes=4 , devices = ["master","slave1","slave2","slave3"]):
    device_frames = []
    for device in devices:
                
        # read in number of frames from metadata idx binary
        data_path = glob.glob(os.path.join(path_to_data,"**","%s*idx.bin"%device), recursive=True) 
        n_frames = 0
        for path in data_path:
            with open(path, mode='rb') as file: # b is important -> binary
                idx_binary = file.read()
                n_frames+= np.frombuffer(idx_binary[16:16+8],dtype=np.uint64)
        
        # read in binary data
        data_path = glob.glob(os.path.join(path_to_data,"**","%s*data.bin"%device), recursive=True)
        data_binary = b""
        for path in data_path:
            with open(path, mode='rb') as file: # b is important -> binary
                byte_data = file.read()
                data_binary += byte_data
                
        
        adc_data = np.frombuffer(data_binary, dtype=np.uint16) # int16 2 bytes per sample
        frame_length = nChirps*nSamples*nChannels*2
        n_frames = len(adc_data)//frame_length
        adc_data = adc_data - ( adc_data >=2^15)* 2^16 # adjust for 12 bit adc data in 16 bit 
        
        frame_array = np.zeros((n_frames,nSamples,nChirps,nChannels),dtype=np.complex64)    
        for i in range(n_frames):
            frame_array[i] = sort_data(adc_data[i*frame_length:(i+1)*frame_length],nSamples,nChirps,nChannels,2,num_lanes)
        device_frames.append(frame_array)
        print("Loaded device: ",device, " with data size: ", frame_array.shape)

    
    print()
    
    frame_array = np.concatenate((device_frames[0],device_frames[1],device_frames[2],device_frames[3]),-1)
    return frame_array
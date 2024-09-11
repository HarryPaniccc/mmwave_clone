import numpy as np 
import cv2
from utils.hdf5.depth import depthHDF5
from utils.hdf5.camera import cameraHDF5
from utils.hdf5.radar import radarHDF5
from utils.npy.depth import depthNPY
from time import sleep

from ultralytics import YOLO
from utils.yolo.segment import segment

import json
import os
import matplotlib.pyplot as plt
# from cv_bridge import CvBridge, CvBridgeError



def main():
    
    working_dir = os.path.dirname(__file__)
    
    yolo_path = os.path.join(working_dir,"utils","yolo","yolov8m-seg.pt")
    yolo_model = YOLO(yolo_path)
    
    f = open(os.path.join(working_dir,'pipeline_config.json'))
    pipeline_config = json.load(f)
    range_pad = pipeline_config["RANGE_CONFIG"]["RANGE_PAD"]
    stereo_intrinsics = pipeline_config["STEREO_CONFIG"]
    data_dir_path = pipeline_config["FILE_PATH"]
    radar_file_name = "Radar_Data_Exp2.hdf5"
    depth_file_name = "Depth_Data_Exp2.hdf5"
    colour_file_name = "Colour_Data_Exp2.hdf5"
    

    ## Open radar data file
    working_dir = os.path.dirname(__file__)
    radar_hdf5 = radarHDF5(os.path.join(working_dir,data_dir_path,radar_file_name))
    samples = radar_hdf5.nSamples
    range_max = radar_hdf5.range_max
    radar_hdf5.display_radar_performance()
    print(radar_hdf5.file["Comments"]["experiment_setup"][0][0].decode())


    total_bins = range_pad + samples
    distances = np.linspace(0,range_max,total_bins)
    bins = np.flip(np.arange(total_bins))
    
    # cv2.namedWindow('Depth Map')
    # cv2.setMouseCallback('Depth Map', click_event)
    
    ## Open depth data file
    print("LOADING DEPTH MAP DATA:")
    depth_cam = depthHDF5(os.path.join(working_dir,data_dir_path,depth_file_name))
    number_of_frames = depth_cam.get_frame_numbers()[1]
    print("Done.\n")
    
    
    print("LOADING COLOUR MAP DATA:")
    colour_cam = cameraHDF5(os.path.join(working_dir,data_dir_path,colour_file_name))
    number_of_frames = depth_cam.get_frame_numbers()[1]
    print("Done.\n")
    
    _, start_time = depth_cam.get_frame(0)
    
    range_bin_seq = []
    distance_over_time = []

    colour_img = None
    selection_map = None
    for i in range(number_of_frames):
        try:
            i+=10
            depth_img, timestamp = depth_cam.get_frame(i)
            colour_img, _ = colour_cam.get_frame(i)
            # colour_img = cv2.cvtColor(colour_img,cv2.COLOR_RGB2BGR)
        except:
            break
        
        selection_map,data_points = segment(colour_img,yolo_model,16) # 16 is the class id for dog in yolov8
        data = depth_img*selection_map

        # calculate moments of binary image
        M = cv2.moments(data)
        
        # calculate x,y coordinate of center
        c_col = int(M["m10"] / M["m00"])
        c_row = int(M["m01"] / M["m00"])
        
        centroid_value = data[c_row,c_col]

        try:
            
            target_bin = bins[np.where(distances<=centroid_value/1000)[0][-1]]
            range_bin_seq.append([timestamp, np.clip(target_bin,0,total_bins-1)])
            distance_over_time.append([timestamp, centroid_value/1000])

            print("Progress: %.2f%%\t | depth: %.2f [m] @%.2fs" % (i*100/number_of_frames,centroid_value/1000 ,timestamp-start_time),end="\r")



            z = data[c_row][c_col]
            x = (c_col - stereo_intrinsics["CX_DEPTH"]) * z / stereo_intrinsics["FX_DEPTH"]
            y = (c_row - stereo_intrinsics["CY_DEPTH"]) * z / stereo_intrinsics["FY_DEPTH"]

            gray_scale_img = cv2.cvtColor(colour_img,cv2.COLOR_RGB2GRAY)
            
            depth_img = depth_img*4 # make it brighter
            depth_image = np.array(depth_img>>8,dtype=np.uint8)
            
            overlay = cv2.fillPoly(colour_img.copy(), np.array([data_points]),(0,0,255))
            weighted_image = cv2.addWeighted(colour_img,0.5,overlay,0.5,0)
            cv2.line(weighted_image, (c_col-10, c_row+10), (c_col+10, c_row-10), (255, 0, 0), 3) 
            cv2.line(weighted_image, (c_col-10, c_row-10), (c_col+10, c_row+10), (255, 0, 0), 3) 
            cv2.putText(weighted_image,"Depth at centroid %.2f m" % (z/1000),(c_col-50, c_row+25),1,1,(255,0,0),2)

            # plt.subplot(221)
            # plt.gca().set_xticks([])
            # plt.gca().set_yticks([])
            # plt.imshow(colour_img[300:500,400:800]) #
            # plt.title("Colour Image")
            
            
            # plt.subplot(222)
            # plt.gca().set_xticks([])
            # plt.gca().set_yticks([])
            # plt.imshow(weighted_image[300:500,400:800])
            # plt.title("YOLO Segmented Colour Image with Centroid")
            
            
            # plt.subplot(223)
            # depth_image = cv2.cvtColor(depth_image,cv2.COLOR_GRAY2BGR)
            # plt.gca().set_xticks([])
            # plt.gca().set_yticks([])
            # plt.imshow(depth_image[300:500,400:800])
            # plt.title("Depth Image")
            
            # plt.subplot(224)
            # plt.gca().set_xticks([])
            # plt.gca().set_yticks([])
            # selection_map = np.array(selection_map,dtype=np.uint8)
            # masked_depth_map = cv2.bitwise_and(depth_image,depth_image,mask = selection_map)
            # cv2.line(masked_depth_map, (c_col-10, c_row+10), (c_col+10, c_row-10), (255, 0, 0), 3) 
            # cv2.line(masked_depth_map, (c_col-10, c_row-10), (c_col+10, c_row+10), (255, 0, 0), 3) 
            # cv2.putText(masked_depth_map,"Depth %.2f m" % (z/1000),(c_col-50, c_row+25),1,1,(255,0,0),2)
            # plt.imshow(masked_depth_map[300:500,400:800],cmap="gray")
            # plt.title("Segment Masked Depth Image with Centroid")
            
            # plt.show()
            # exit()
            
            scale_percent = 80 # percent of original size
            width = int(weighted_image.shape[1] * scale_percent / 100)
            height = int(weighted_image.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            # resize image
            resized = cv2.resize(weighted_image, dim, interpolation = cv2.INTER_AREA)
            
            # convert type

            cv2.imshow('Segmented Image with Depth at Centroid',resized)
            if cv2.waitKey(1) == ord('p'):
                 cv2.waitKey(-1)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        except KeyboardInterrupt:
            print()
            break
        
    print("\n\n\n")
    print("Saving.\n")

    np.save(os.path.join(working_dir,data_dir_path,"rs_bins"), range_bin_seq)
    np.save(os.path.join(working_dir,data_dir_path,"rs_distance"), distance_over_time)

    print("Done.")

if __name__ == "__main__":
    main()
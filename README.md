<p float="left">
  <img src="docs/resources/ARU_logo_rectangle.png" width="600" />
  <img src="docs/resources/rrsglogo.png" width="150" /> 
</p>

# MMWAVE-POST-PROCESS

## Author
Nicholas Bowden - UCT MSc

## Description 
Pipeline for handing and processing mmWave Data. The pipeline uses data of a specific format. This format is created by code in: [hdf5 download](https://github.com/RRSG-mmWave/ECAL-TO-HDF5)   

## Usage
Whatever script you are running needs to be in the correct directory. Keep the following directory structure:
```tree
|--MMWAVE-POST-PROCESS/
    |--data/
        |--experiment_directory_0/
            |-- files
        |--experiment_directory_1/
            |-- files
        |-- etc.

    |--utils/
    |--pipeline_config.json
    |--script_being_run.py

```   
Effectively the MMWAVE-POST-PROCESS directory is your working directory. I've left the usage of this repo pretty open ended because everyone will have different requirements. However, the utils folder contains classes and functions you should be able to use as building blocks for you code. 

### Config File
Edit the pipeline config json file to specify the data directory and other parameters you want to be kept constant through the pipeline. You can add to this file as needed. There are plenty of example scripts and the main scripts that show how to use it. 

### MMWAVE 1: Depth Camera
The mmwave_1_camera_depth.py script uses depth data in an hdf5 format and known radar parameters to calculate the range bin over time of a target visually. Just run the script and click on the point you want to track. If the target is moving you will need to click continuosly.   

Missing Feature: I would recommend adding in the usage of colour camera data and a pretrained segmentation AI like DINO to find out what pixels in the depth camera image correspond to your target automatically. If you do this approach you will need to calibrate the colour camera to the depth image.  

### MMWAVE 2: Radar Phase
You shouldn't need to fiddle with this script too much. Just run it and look at the phase and see if its what you expect. 

### MMWAVE 3a: Spectrogram
Shows an example of how to filter and plot spectrogram. Adjust the code as needed.

### MMWAVE 3b: DNN Data Creator
Shows an example of how to generate phase time series data with a sliding window..


## Additional Scripts Directory
In the additional scripts directory you will find more example code aimed at various purposes including generic radar processing under examples, some deep learning with pytorch (under deep learning) and most importantly some code on how to handle MMWCAS .bin files. 

### MMWCAS
The cascaded radar system works exactly the same in the pipeline as another FMCW Radar data from the AWR1843 (and probably other radars). However, the raw data is captured over 4 LVDS lanes not 2 LVDS lanes and is therefore packaged differently. After the data is sorted though, the FMCW radar cude is exactly the same but just with more virtual receive channels so the pipeline will work the same for MMWCAS radar data and xWRxxxx radar data. Scripts for generating the radar data cube is found in the mmwcas directory in additional scripts. 

## Conclusion
Have fun... 
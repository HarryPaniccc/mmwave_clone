import cv2
import numpy as np

def segment(img,yolo_model,class_id):
    
    results = yolo_model(source=img.copy(),verbose=False)
    gray_scale = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    selection = np.zeros(gray_scale.shape)
    for i, box in enumerate(results[0].boxes):
        detected_class = int(box.cls.cpu())
        if results[0].masks:
            points_rollers = results[0].masks[i].xy[0].astype(int)

            # # names: {0: 'mass', 1: 'filter', 2: 'splitter', 3: 'rollers'}
            if detected_class == class_id:
                data_points = [tuple(elem) for elem in points_rollers]
                cv2.fillPoly(selection, np.array([data_points]),1)
                
    return selection, data_points
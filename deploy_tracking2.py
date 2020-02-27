from fastai.vision import *
from fastai import *
from fastai.metrics import error_rate
import cv2
from keras.models import load_model
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model1 = load_model('baseline_original_all_epochs.h5')
model2 = load_learner('./models')

def model_prediction(frame, model1, model2):
    
    try:
        # set threshold to 0.4 based on ROC-AUC curved at modelling stage
        threshold = 0.4
        
        # extract target heigh and weight from the model 1 (Keras Model)
        target_height = model1.input_shape[1]
        target_width = model1.input_shape[2]
        
        # convert back to original RGB color channel from OpenCV's default of BGR
        color_corrected_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # resize model
        resized_image = cv2.resize(color_corrected_frame, (target_height, target_width))
        
        # expand dimensions to fit model
        expanded_image = np.expand_dims(resized_image, axis = 0)
        
        # rescale pixels to be between 0 and 1
        rescaled_image = expanded_image / 255

        prediction_prob_1 = round(model1.predict_proba(rescaled_image)[0][0], 2)
        prediction_outcome_1 = None

        if prediction_prob_1 >= threshold:
            prediction_outcome_1 = 'Recyclable'
        else:
            prediction_outcome_1 = 'Organic'
        
        if prediction_outcome_1 == 'Organic':
            return (prediction_outcome_1, 1 - prediction_prob_1)
        
        elif prediction_outcome_1 == 'Recyclable':
            img_fastai = Image(pil2tensor(color_corrected_frame, dtype=np.float32).div_(255))
            fastai_prediction = model2.predict(img_fastai)
            prediction_outcome_2 = str(fastai_prediction[0])
            prediction_prob_2 = np.array(fastai_prediction[2]).max()
            return (prediction_outcome_1, prediction_prob_1, prediction_outcome_2, prediction_prob_2)
        
    except cv2.error:
        prediction_outcome_1 = 'Frame error'
        return (prediction_outcome_1)

fontFace = cv2.FONT_HERSHEY_SIMPLEX
lineType = cv2.LINE_AA
    
    
tracker = cv2.TrackerKCF_create()
video = cv2.VideoCapture(0)
ok, frame = video.read()

bbox = (287, 23, 86, 320)
bbox = cv2.selectROI("Tracking", frame, True)
ok = tracker.init(frame, bbox)

while True:
    ok, frame = video.read()
    
    timer = cv2.getTickCount()
    
    ok, bbox = tracker.update(frame)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        
        cv2.rectangle(frame, p1, p2, (0, 0, 255), 3, 2)
        rect_img = frame[p1[1] : p2[1], p1[0] : p2[0]]
     
        prediction_output = model_prediction(rect_img, model1, model2)
         
        if p2[0] > 1279 or p2[1] < 1 or p2[1] > 719 or p1[0] < 1 or p1[1] < 1:
            cv2.putText(frame, "Outcome: Frames out of bound", (50,500), fontFace, 1, (0, 0, 255), 2, lineType)
            
        elif prediction_output[0] == 'Recyclable':
            cv2.putText(frame, "FPS : " + str(int(fps)), (50,500), fontFace, 1, (0, 255, 0), 2, lineType)
            cv2.putText(frame, "Outcome 1: " + prediction_output[0], (50,550), fontFace, 1, (0, 255, 0), 2, lineType)
            cv2.putText(frame, "Prob. 1: " + str(prediction_output[1]), (50,600), fontFace, 1, (0, 255, 0), 2, lineType)
            cv2.putText(frame, "Outcome 2: " + prediction_output[2], (50,650), fontFace, 1, (0, 255, 0), 2, lineType)
            cv2.putText(frame, "Prob. 2: " + str(prediction_output[3]), (50,700), fontFace, 1, (0, 255, 0), 2, lineType)

        elif prediction_output[0] == 'Organic':
            cv2.putText(frame, "FPS : " + str(int(fps)), (50,500), fontFace, 1, (0, 255, 0), 2, lineType)
            cv2.putText(frame, "Outcome 1: " + prediction_output[0], (50,550), fontFace, 1, (0, 255, 0), 2, lineType)
            cv2.putText(frame, "Prob. 1: " + str(prediction_output[1]), (50,600), fontFace, 1, (0, 255, 0), 2, lineType)
        
    else:
        cv2.putText(frame, "Tracking failure detected", (50,450), fontFace, 1, (0, 0, 255), 2, lineType)
        
    cv2.imshow("Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
video.release()
cv2.destroyAllWindows()
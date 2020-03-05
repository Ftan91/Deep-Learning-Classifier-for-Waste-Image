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
            return (prediction_outcome_1, round(1 - prediction_prob_1, 3))
        
        elif prediction_outcome_1 == 'Recyclable':
            img_fastai = Image(pil2tensor(color_corrected_frame, dtype=np.float32).div_(255))
            fastai_prediction = model2.predict(img_fastai)
            prediction_outcome_2 = str(fastai_prediction[0])
            prediction_prob_2 = np.array(fastai_prediction[2]).max()
            return (prediction_outcome_1, round(prediction_prob_1, 3), prediction_outcome_2, round(prediction_prob_2, 3))
        
    except cv2.error:
        prediction_outcome_1 = 'Frame error'
        return (prediction_outcome_1)

font = cv2.FONT_HERSHEY_SIMPLEX
size = 0.5
line = cv2.LINE_AA
  
frame = cv2.imread('test_image3.png')

bboxes = []
colors = []

while True:
    bbox = cv2.selectROI('Classifier', frame)

    bboxes.append(bbox)
    colors.append((np.random.randint(0,255), np.random.randint(0, 255), np.random.randint(0, 255)))
    
    print ("Press q to quit selecting boxes and start tracking")
    print ("Press any other key to select next object")
    
    k = cv2.waitKey(0) & 0xFF
    if (k == 113):
        break
        
print ('Selected bounding boxes {}'.format(bboxes))

while True:

    frame = cv2.imread('test_image3.png')
    
    for i, box in enumerate(bboxes):
        
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 3, 1)
        
        rect_img = frame[p1[1] : p2[1], p1[0] : p2[0]]

        prediction_output = model_prediction(rect_img, model1, model2)

        Out1_loc = (p1[0], p1[1] - 50)
        Prob1_loc = (p1[0], p1[1] - 35)
        Out2_loc = (p1[0], p1[1] - 20)
        Prob2_loc = (p1[0], p1[1] - 5)

        if prediction_output[0] == 'Recyclable':
            cv2.putText(frame, "Outcome 1: " + prediction_output[0], Out1_loc, font, size, colors[i], 2, line)
            cv2.putText(frame, "Prob. 1: " + str(prediction_output[1]), Prob1_loc, font, size, colors[i], 2, line)
            cv2.putText(frame, "Outcome 2: " + prediction_output[2], Out2_loc, font, size, colors[i], 2, line)
            cv2.putText(frame, "Prob. 2: " + str(prediction_output[3]), Prob2_loc, font, size, colors[i], 2, line)

        elif prediction_output[0] == 'Organic':
            cv2.putText(frame, "Outcome 1: " + prediction_output[0], Out2_loc, font, size, colors[i], 2, line)
            cv2.putText(frame, "Prob. 1: " + str(prediction_output[1]), Prob2_loc, font, size, colors[i], 2, line)
        
    cv2.imshow('Classifier', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
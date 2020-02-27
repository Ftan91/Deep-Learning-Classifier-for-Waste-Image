import cv2
from keras.models import load_model
import numpy as np

model1 = load_model('baseline_original_all_epochs.h5')

def model_prediction(rect_img, model1):
    
    try:
        threshold = 0.4
        
        target_height1 = model1.input_shape[1]
        target_width1 = model1.input_shape[2]
        
        color_corrected = cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB)
        resize_model1 = cv2.resize(color_corrected,(target_height1, target_width1))
        resize_model1 = np.expand_dims(resize_model1, axis = 0)
        resize_model1 = resize_model1 / 255

        prediction_model1 = round(model1.predict_proba(resize_model1)[0][0], 2)
        prediction_outcome = None

        if prediction_model1 > threshold:
            prediction_outcome = 'Recyclable'
        else:
            prediction_outcome = 'Organic'
        
    except cv2.error:
        prediction_model1 = 0
        prediction_outcome = 'Unable to extract frame. Select another frame.'
    
    return (prediction_outcome, prediction_model1)
  
def draw_rectangle(event,x,y,flags,param):

    global pt1, pt2, topLeft_clicked,botRight_clicked

    if event == cv2.EVENT_LBUTTONDOWN:

        if topLeft_clicked == True and botRight_clicked == True:
            topLeft_clicked = False
            botRight_clicked = False
            pt1 = (0,0)
            pt2 = (0,0)

        if topLeft_clicked == False:
            pt1 = (x,y)
            topLeft_clicked = True
            
        elif botRight_clicked == False:
            pt2 = (x,y)
            botRight_clicked = True

pt1 = (0,0) # upper left (or first clicked point)
pt2 = (0,0) # bottom right (or second clicked point)
topLeft_clicked = False
botRight_clicked = False

cap = cv2.VideoCapture(0)

cv2.namedWindow('Classifier')
cv2.setMouseCallback('Classifier', draw_rectangle) 

while True:

    ret, frame = cap.read()

    if topLeft_clicked:
        cv2.circle(frame, center = pt1, radius=5, color=(0, 0, 255), thickness=-1)

    if topLeft_clicked and botRight_clicked:
        cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
        
        if (pt2[0] > pt1[0]) and (pt2[1] > pt1[1]):
            rect_img = frame[pt1[1] : pt2[1], pt1[0] : pt2[0]]
            org = (pt1[0] + 5, pt1[1] - 5)
        elif (pt2[0] < pt1[0]) and (pt2[1] < pt1[1]):
            rect_img = frame[pt2[1] : pt1[1], pt2[0] : pt1[0]]
            org = (pt2[0] + 5, pt2[1] - 5)
        elif (pt2[0] > pt1[0]) and (pt2[1] < pt1[1]):
            rect_img = frame[pt2[1] : pt1[1], pt1[0] : pt2[0]]
            org = (pt1[0] + 5, pt2[1] - 5)
        elif (pt2[0] < pt1[0] and (pt2[1] > pt1[1])):
            rect_img = frame[pt1[1] : pt2[1], pt2[0] : pt1[0]]
            org = (pt2[0] + 5, pt1[1] - 5)
       
        model_output = model_prediction(rect_img, model1)
        cv2.putText(frame, '{} : {}'.format(model_output[0], model_output[1]), org, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
    cv2.imshow('Classifier', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
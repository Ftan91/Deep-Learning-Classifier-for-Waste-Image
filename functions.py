import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_images(num_row, num_col, image_list, selected_photos = [0, 1, 2, 3, 4, 5]):
    """Plot images"""
    try:
        fig, axes = plt.subplots(num_row, num_col, figsize = (18,10))
        count = 0
        for row in range(num_row):
            for col in range(num_col):
                axes[row][col].imshow(image_list[selected_photos[count]])
                count += 1
    
    except IndexError:
        print ('Number of selected photos not aligned with number of available cells: {}'.format(num_row * num_col))
        
def predict(image, model, outcome_dict):
    """image should be an array"""
    target_height = model.input_shape[1]
    target_width = model.input_shape[2]
    
    resized_img = cv2.resize(image,(target_width, target_height))
    resized_img = np.expand_dims(resized_img, axis = 0)
    resized_img = resized_img / 255
    
    predict = model.predict_classes(resized_img)[0][0]
    outcome = outcome_dict[predict]
    predict_proba = round(model.predict(resized_img)[0][0],5)
    
    print ('Prediction: {} \nProbability: {}'.format(outcome, predict_proba))
    
    plt.imshow(image)
    plt.show()
    
def plot_confusion_matrix(y_true, y_pred, positive_label, negative_label):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    con_matrix = confusion_matrix(y_true, y_pred)
    cm_list = con_matrix.ravel()
    cm_list[1], cm_list[2] = cm_list[2], cm_list[1]
    cm_string = ['True Negatives:', 'False Negatives:', 'False Positives:', 'True Positives:']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(con_matrix, cmap = plt.cm.OrRd)
    plt.grid(b = None)
    fig.colorbar(cax)

    count = 0
    for i in range(2):
        for j in range(2):
            if cm_string[count].split()[0] == 'True':
                plt.text(i, j, cm_string[count] + '\n' + str(cm_list[count]), va = 'center', ha = 'center', color = 'black')
                count +=1
            else:
                plt.text(i, j, cm_string[count] + '\n' + str(cm_list[count]), va = 'center', ha = 'center', color = 'red')
                count +=1
    labels = [0,1]

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(labels, [negative_label, positive_label])
    plt.yticks(labels, [negative_label, positive_label])
    plt.show()
# Deep-Learning-Classifier-for-Waste-Image

Link to slides [here](https://docs.google.com/presentation/d/1_zzVJjtJ5vg5u0akq-QbtEjLUtVgq1u4IIIbaHyEiQM/edit?usp=sharing).

### **Problem Statement**
Given the recent headlines around South East Asian countries sending back their trash to developed nations, I've thought of tinkering around with some solutions in the spirit of creating a low-cost AI solution, particularly for waste management firms, to make the recycling process more efficient and simultaneously aleviate the margin pressure of recyclable products.

![Waste](images/world_map.png)
Source: Verisk Maplecroft, 2019

### **Solution**
As such, the plan  is to construct deep learning model (convolutional neural network) to classify waste images firstly by distinguishing if they are a recyclable or organic item and subsequently classifying them into its individual counterparts such as paper, glass, plastic, cardboard and metal.

As a teaser, final output looks something like this:

[![](http://img.youtube.com/vi/PdsfRcfIfBQ/0.jpg)](http://www.youtube.com/watch?v=PdsfRcfIfBQ "Image Classifier using Convolutional Neural Network")


### **Dataset**
**Model 1.** For the first model, dataset can be obtained from Kaggle [here](https://www.kaggle.com/techsash/waste-classification-data). The dataset contains of 22,564 images pre-labeled as 'Organic' and 'Recyclable' items and has been split into 85% train and 15% test. 

**Model 2.** With regards to the second model, the dataset is obtainable [here](https://github.com/garythung/trashnet) with credits given to Gary Thung and Mindy Yang, as images has been painstakingly gathered manually for their final project in Standford's Computer Science 229: Machine Learning class. Hats off to both of them again. The dataset spans across six recyclable classes notably:

- 501 glass
- 594 paper
- 403 cardboard
- 482 plastic
- 410 metal
- 137 trash [will be removed for now as this is not much of a value add to my final model]

### **Setting up Cloud Infrastructure**
In light of my limited computational resources, it is best to set up an account with Google Cloud Platform, for which you will be granted c. £231 in free credit (thanks Google!). For low level task, I've ran my notebook purely on 4vCPU with 15GB memory and for training my CNN, I've also added on a NVIDIA Tesla P100 GPU for rapid processing. FYI - to be granted access to a GPU, you will need to request access (approvals takes less than an hour). Only advise is to make sure you shutdown the virtual instance for which you are running the notebook in, unless you are keen to be charged for idle time!

There are many useful instructions to get GCP up and running. One I found useful is this video tutorial [here](https://www.youtube.com/watch?v=Db4FfhXDYS8).

### **Folder Organization [Important]**
To avoid overloading images into self-created X & Y variables which eventually causes you to run out of memory, I instead used Keras's ImageDataGenerator (can add augmentation to images as well) function which pretty much reads images directly from the images directory. In order to do this however, we will need the folders to be correctly organized under the format of:

    TRAIN
        - R 
        - O
    VALIDATION
        - R
        - O
    TEST
        - R
        - O
        
It is up to you if you would like a validation set or not but more importantly each subfolder within the parent folder needs to be correctly labeled under the appropriate class, in this case being 'R' for Recyclable and 'O' for Organic. Also be aware of any hidden files such as .ipynbcheckpoints whenever moving files around. To speed up the process of moving files around, you can utilize the Python's shutil modules which offers a number of high-level operations on files and collection of files. 

### **Image Exploration [EDA]**
With the first dataset, some issues that may be pose issues down the road:

- pictures may include elements of both organic and recyclable
- clean pictures (not a reflection of reality) vs noisy background
- some images are incorrectly labelled across both labels
- some images represent posters / flyers of labelled class and as such is not very clear cut

![Recyclable](images/recyclable.png)
![Organic](images/organic.png)


### **Model Performance**

### **Model Deployment & Demo**

### **Limitations, Lessons and Future Work**
Data Image Augmentation
Pre-processing image, applying gaussian blur to reduce noise particularly for noisy background images

### **References and Acknowledgement**

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required packages
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import tensorflow
import tensorflow.keras as keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,BatchNormalization
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import timeit


# ## 1. Load the datasets

# For the project, we provide a training set with 50000 images in the directory `../data/images/` with:
# - noisy labels for all images provided in `../data/noisy_label.csv`;
# - clean labels for the first 10000 images provided in `../data/clean_labels.csv`. 

# In[2]:


# [DO NOT MODIFY THIS CELL]

# load the images
n_img = 50000
n_noisy = 40000
n_clean_noisy = n_img - n_noisy
imgs = np.empty((n_img,32,32,3))
for i in range(n_img):
    img_fn = f'../data/images/{i+1:05d}.png'
    imgs[i,:,:,:]=cv2.cvtColor(cv2.imread(img_fn),cv2.COLOR_BGR2RGB)

# load the labels
clean_labels = np.genfromtxt('../data/clean_labels.csv', delimiter=',', dtype="int8")
noisy_labels = np.genfromtxt('../data/noisy_labels.csv', delimiter=',', dtype="int8")


# For illustration, we present a small subset (of size 8) of the images with their clean and noisy labels in `clean_noisy_trainset`. You are encouraged to explore more characteristics of the label noises on the whole dataset. 

# In[3]:


# [DO NOT MODIFY THIS CELL]

fig = plt.figure()

ax1 = fig.add_subplot(2,4,1)
ax1.imshow(imgs[0]/255)
ax2 = fig.add_subplot(2,4,2)
ax2.imshow(imgs[1]/255)
ax3 = fig.add_subplot(2,4,3)
ax3.imshow(imgs[2]/255)
ax4 = fig.add_subplot(2,4,4)
ax4.imshow(imgs[3]/255)
ax1 = fig.add_subplot(2,4,5)
ax1.imshow(imgs[4]/255)
ax2 = fig.add_subplot(2,4,6)
ax2.imshow(imgs[5]/255)
ax3 = fig.add_subplot(2,4,7)
ax3.imshow(imgs[6]/255)
ax4 = fig.add_subplot(2,4,8)
ax4.imshow(imgs[7]/255)

# The class-label correspondence
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# print clean labels
print('Clean labels:')
print(' '.join('%5s' % classes[clean_labels[j]] for j in range(8)))
# print noisy labels
print('Noisy labels:')
print(' '.join('%5s' % classes[noisy_labels[j]] for j in range(8)))


# ## 2. The predictive model
# 
# We consider a baseline model directly on the noisy dataset without any label corrections. RGB histogram features are extracted to fit a logistic regression model.

# ### 2.1. Baseline Model

# In[4]:


# [DO NOT MODIFY THIS CELL]
# RGB histogram dataset construction
no_bins = 6
bins = np.linspace(0,255,no_bins) # the range of the rgb histogram
target_vec = np.empty(n_img)
feature_mtx = np.empty((n_img,3*(len(bins)-1)))
i = 0
for i in range(n_img):
    # The target vector consists of noisy labels
    target_vec[i] = noisy_labels[i]
    
    # Use the numbers of pixels in each bin for all three channels as the features
    feature1 = np.histogram(imgs[i][:,:,0],bins=bins)[0] 
    feature2 = np.histogram(imgs[i][:,:,1],bins=bins)[0]
    feature3 = np.histogram(imgs[i][:,:,2],bins=bins)[0]
    
    # Concatenate three features
    feature_mtx[i,] = np.concatenate((feature1, feature2, feature3), axis=None)
    i += 1


# In[5]:


# [DO NOT MODIFY THIS CELL]
# Train a logistic regression model 
clf = LogisticRegression(random_state=0).fit(feature_mtx, target_vec)


# For the convenience of evaluation, we write the following function `predictive_model` that does the label prediction. **For your predictive model, feel free to modify the function, but make sure the function takes an RGB image of numpy.array format with dimension $32\times32\times3$  as input, and returns one single label as output.**

# In[6]:


# [DO NOT MODIFY THIS CELL]
def baseline_model(image):
    '''
    This is the baseline predictive model that takes in the image and returns a label prediction
    '''
    feature1 = np.histogram(image[:,:,0],bins=bins)[0]
    feature2 = np.histogram(image[:,:,1],bins=bins)[0]
    feature3 = np.histogram(image[:,:,2],bins=bins)[0]
    feature = np.concatenate((feature1, feature2, feature3), axis=None).reshape(1,-1)
    return clf.predict(feature)


# ### 2.2. Model I

# In[7]:


# train_valid_test split
idx=list(range(10000))
test_idx=np.array(random.sample(idx,2000))
train_idx=np.delete(np.array(range(10000)),test_idx)
imgs_train = imgs[np.append(train_idx,np.array(range(10000,50000)))]
imgs_test = imgs[test_idx]
labels_train=np.append(clean_labels[train_idx],noisy_labels[10000:50000])
labels_test=clean_labels[test_idx]
# Normalize x
X_train = np.array(imgs_train) / 255
X_test = np.array(imgs_test) / 255


# In[8]:


# [BUILD A MORE SOPHISTICATED PREDICTIVE MODEL]

# CNN
def model_I(image):
    '''
    This function should takes in the image of dimension 32*32*3 as input and returns a label prediction
    '''
    #create model
    model = Sequential()
    #add model layers
    model.add(Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10))
    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    #train the model
    history = model.fit(X_train, labels_train, epochs=10)
    #predict
    X_test = np.array(image)/255
    label = model.predict(X_test)
    model.save('model1.h5')
    label = np.argmax(np.round(label), axis=1)
    return label


# In[9]:


# test for CNN (less than 10 min)
start = timeit.default_timer()
labels_pred = model_I(imgs_test)
stop = timeit.default_timer()
print('Time: ', stop - start, 'seconds')


# ### 2.3. Model II

# In[10]:


# [ADD WEAKLY SUPERVISED LEARNING FEATURE TO MODEL I]

#CNN
def model_II(image):
    #create model
    model = Sequential()
    #add model layers
    model.add(Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10))
    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    #train the model
    history = model.fit(X_train_2, labels_train, epochs=10)
    #predict
    X_test = np.array(image)/255    
    label = model.predict(X_test)
    model.save('model2.h5')
    label = np.argmax(np.round(label), axis=1)
    return label


# In[20]:


start = timeit.default_timer()

# train CNN using clean labels
imgs_train_2, imgs_test_2, labels_train, labels_test  = train_test_split(imgs[0:10000], clean_labels, test_size=0.1, random_state=7)
# Normalize x
X_train_2 = np.array(imgs_train_2) / 255
X_test_2 = np.array(imgs_test_2) / 255
label_pred1=model_II(imgs)

# find images with consistent predicted label and noisy label, and set them as clean label too to enlarge the size of training data
new_clean_idx=np.append(range(10000),np.array(range(10000,50000))[label_pred1[10000:]==noisy_labels[10000:]])
new_clean_labels=np.append(clean_labels,noisy_labels[new_clean_idx[10000:]])
new_clean_imgs=imgs[new_clean_idx]

# train CNN using updated clean labels
new_idx=np.array(range(new_clean_idx.shape[0]))
test_idx=np.array(random.sample(list(range(10000)),2000))
train_idx=np.delete(new_idx,test_idx)
imgs_train_2 = new_clean_imgs[train_idx]
imgs_test_2 = new_clean_imgs[test_idx]
labels_train = new_clean_labels[train_idx]
labels_test = new_clean_labels[test_idx]
# Normalize x
X_train_2 = np.array(imgs_train_2) / 255
X_test_2 = np.array(imgs_test_2) / 255
label_pred2=model_II(imgs_test_2)

acc=np.mean(label_pred2==labels_test)
print('The accuracy of the cnn in the test dataset is: %3f'%(acc))

import pandas as pd
dt=pd.DataFrame(label_pred2,labels_test)
dt.to_csv('dt.csv')

stop = timeit.default_timer()
print('Time: ', stop - start, 'seconds')


# ## 3. Evaluation

# For assessment, we will evaluate your final model on a hidden test dataset with clean labels by the `evaluation` function defined as follows. Although you will not have the access to the test set, the function would be useful for the model developments. For example, you can split the small training set, using one portion for weakly supervised learning and the other for validation purpose. 

# In[ ]:


# [DO NOT MODIFY THIS CELL]
def evaluation(model, test_labels, test_imgs):
    y_true = test_labels
    y_pred = []
    for image in test_imgs:
        y_pred.append(model(image))
    print(classification_report(y_true, y_pred))


# In[ ]:


# [DO NOT MODIFY THIS CELL]
# This is the code for evaluating the prediction performance on a testset
# You will get an error if running this cell, as you do not have the testset
# Nonetheless, you can create your own validation set to run the evlauation
n_test = 10000
test_labels = np.genfromtxt('../data/test_labels.csv', delimiter=',', dtype="int8")
test_imgs = np.empty((n_test,32,32,3))
for i in range(n_test):
    img_fn = f'../data/test_images/test{i+1:05d}.png'
    test_imgs[i,:,:,:]=cv2.cvtColor(cv2.imread(img_fn),cv2.COLOR_BGR2RGB)
evaluation(baseline_model, test_labels, test_imgs)


# In[ ]:


#Evaluation of model I
start = timeit.default_timer()
model1 = keras.models.load_model('model1.h5')
evaluation(model1, test_labels, test_imgs)
stop = timeit.default_timer()
print('Time: ', stop - start, 'seconds')

#Evaluation of model II
start = timeit.default_timer()
model2 = keras.models.load_model('model2.h5')
evaluation(model2, test_labels, test_imgs)
stop = timeit.default_timer()
print('Time: ', stop - start, 'seconds')


# The overall accuracy is $0.24$, which is better than random guess (which should have a accuracy around $0.10$). For the project, you should try to improve the performance by the following strategies:
# 
# - Consider a better choice of model architectures, hyperparameters, or training scheme for the predictive model;
# - Use both `clean_noisy_trainset` and `noisy_trainset` for model training via **weakly supervised learning** methods. One possible solution is to train a "label-correction" model using the former, correct the labels in the latter, and train the final predictive model using the corrected dataset.
# - Apply techniques such as $k$-fold cross validation to avoid overfitting;
# - Any other reasonable strategies.

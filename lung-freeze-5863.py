# from https://github.com/yongsingyou/chest-x-ray/blob/master/pneumonia.ipynb
# data: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
# from keras.applications.inception_v3 import InceptionV3
from keras import applications
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img,img_to_array
from keras import optimizers

import numpy as np 
# pandas is for 
# 1. group label from folder (= using flow_from_directory+class_mode='binary')
# 2. show table on jupyter notebook
import pandas as pd 
import glob
import os
from pathlib import Path

import time

MODEL_VGG = 'vgg'
MODEL_RESNET = 'resnet'

# TODO conver the below to a class
g_model = MODEL_RESNET # default model

# data path
data_dir = Path('./chest_xray/')
train_dir = data_dir / 'train'
val_dir = data_dir / 'val'
test_dir = data_dir / 'test'
train_features_path = 'lung_bottleneck_features_train.npy'
val_features_path = 'lung_bottleneck_features_validation.npy'
top_model_weights_path = 'lung_bottleneck_fc_model.h5'
model_weights_path = 'lung_model.h5'

# offline bottleneck conv features
full_train_feature = "" 
full_val_feature = ""

img_width, img_height = 150,150 # 224, 224 (default imagenet size)
epochs = 10
batch_size = 16
nb_train_samples =  4126 # 1000 (normal) + 3216 train_data.shape[0] 5216
nb_validation_samples = 868 # 294 + 572 # val_data.shape[0] 16
nb_test_samples = 864 # 289+575

# def explore_image():
#     import matplotlib.pyplot as plt 
#     import matplotlib.image as mpimg    
#     ## explore images and labels
#     f_normal = './chest_xray/train/NORMAL/IM-0115-0001.jpeg'
#     f_pneumonia = './chest_xray/train/PNEUMONIA/person1000_bacteria_2931.jpeg'
#     plt.subplot(121)
#     plt.imshow(load_img(f_normal, target_size=(224, 224)))
#     plt.subplot(122)
#     plt.imshow(load_img(f_pneumonia, target_size=(224, 224)))
#     try:
#         plt.show(block=True)
#     except:
#         print("plot exception")    
#     #original = load_img(f_normal, target_size=(224, 224))

def output_dataframe(path):
    '''
    This function produce dataframe of label with shuffling.
    
    input
    path: train, val or test directory
    
    output:
    dataframe for label
    
    '''
    
    n_dir = path / 'NORMAL'
    p_dir = path / 'PNEUMONIA'
    n_img = n_dir.glob('*.jpeg')
    p_img = p_dir.glob('*.jpeg')
    
    data = []

    for img in n_img:
        data.append((img,0))

    for img in p_img:
        data.append((img,1))
    
    # build dataframe
    data = pd.DataFrame(data,
                          columns=['image','label'],index=None)
    #data = data.sample(frac=1.).reset_index(drop=True)
    
    return data

## run image through the model and save feature output
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# The reason why we are storing the features offline rather than adding 
# our fully-connected model directly on top of a frozen convolutional base and 
# running the whole thing, is computational effiency. 
# Running VGG16 is expensive, especially if you're working on CPU, and 
# we want to only do it once. Note that this prevents us from using data augmentation.
def save_bottlebeck_features():    
    datagen = ImageDataGenerator(rescale=1. / 255)
     
    # build the VGG16 network
    if g_model != MODEL_VGG:
        return 

    model = applications.VGG16(include_top=False, weights='imagenet')    

    global full_train_feature, full_val_feature
    full_train_feature = model + "_" + train_features_path
    full_val_feature = model + "_" + val_features_path 

    if os.path.isfile(full_train_feature) and os.path.isfile(full_val_feature):
        return

    # takes 9 min on mackbook pro w/ cpu mode
    generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(full_train_feature,
            bottleneck_features_train)
    print('train_feature done')

    generator = datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(full_val_feature,
            bottleneck_features_validation)
    print('val_feature done')

def load_model(use_self_trained_weight=False):
    model = None
    base_model = None
    if g_model == MODEL_VGG:  
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
        print('base Model loaded.')

        # build a classifier model to put on top of the convolutional model
        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(1, activation='sigmoid'))
        full_top_model_weight = g_model + "_" + top_model_weights_path
        if use_self_trained_weight and os.path.isfile(full_top_model_weight):            
            top_model.load_weights(full_top_model_weight)

        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))  
    elif g_model == MODEL_RESNET:
        # TODO: add resnet 
        # no use save_bottlebeck_features !! need more training time
        base_model = applications.ResNet50(include_top=False, weights='imagenet')
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)

        # and a logistic layer 
        # sigmoid is perfect for a binary classification 
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        predictions = Dense(1, activation='sigmoid')(x)        
        model = Model(inputs=base_model.input, outputs=predictions)

    model_weight = g_model + "_" + model_weights_path
    if use_self_trained_weight and os.path.isfile(model_weight):     
        model.load_weights(model_weight)
    return model, base_model
        
## train top layer
def train_top_model():            
    # if g_model == MODEL_VGG:  
        # get features from frozen conv part  
        # start_time = time.time()
        # save_bottlebeck_features()
        # print(time.time() - start_time, "seconds")

        # train_data = np.load(full_train_feature)
        # train_df = output_dataframe(train_dir) # pandas
        # # train_df.head()    
        # # train_df.label.value_counts()               
        # len_ = (len(train_df.label.values)//batch_size)*batch_size
        # train_labels = train_df.label.values[:len_]  
        
        # validation_data = np.load(full_val_feature)         
        # val_df = output_dataframe(val_dir)       
        # len_ = (len(val_df.label.values)//batch_size)*batch_size
        # validation_labels = val_df.label.values[:len_]                  

        # model = Sequential()
        # model.add(Flatten(input_shape=train_data.shape[1:]))
        # model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(1, activation='sigmoid'))
    # elif g_model == MODEL_RESNET:

    model, base_model = load_model()
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # TODO: add data augmentation
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary') # shuffle is default True
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')        

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    # if g_model == MODEL_VGG:     
    #     train_history = model.fit(train_data, train_labels,
    #             epochs=epochs,
    #             batch_size=batch_size,
    #             validation_data=(validation_data, validation_labels))              
    #     model.save_weights(g_model + "_" + top_model_weights_path)
    # elif g_model == MODEL_RESNET:
    train_history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        verbose=2)        
    model.save_weights(g_model + "_" + model_weights_path)

    from utility import show_train_history
    show_train_history(train_history,'acc','val_acc')     
    show_train_history(train_history,'loss','val_loss')

def evaluate_model():
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    model, _ = load_model(use_self_trained_weight=True)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    scores = model.evaluate_generator(generator, nb_test_samples // batch_size)
    print("loss:{}".format(scores[0])) # 
    print("acc:{}".format(scores[1])) # 

def main():
    # explore_image()   

    global g_model 
    g_model = MODEL_VGG

    train_top_model() # using matplotlib to plot

    evaluate_model()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")


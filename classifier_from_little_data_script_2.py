# modified from https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069#gistcomment-2714226
# This script goes along the blog post
# "Building powerful image classification models using very little data"
# from blog.keras.io.
# It uses data that can be downloaded at:
# https://www.kaggle.com/c/dogs-vs-cats/data
# In our setup, we:
# - created a data/ folder
# - created train/ and validation/ subfolders inside data/
# - created cats/ and dogs/ subfolders inside train/ and validation/
# - put the cat pictures index 0-999 in data/train/cats
# - put the cat pictures index 1000-1399 in data/validation/cats
# - put the dog pictures index 0-999 in data/train/dogs
# - put the dog pictures index 1000-1399 in data/validation/dogs
# So that we have 1000 training examples for each class, and 400 validation examples for each class.
# In summary, this is our directory structure:
# 
# data/
#     train/
#         dogs/
#             dog001.jpg
#             dog002.jpg
#             ...
#         cats/
#             cat001.jpg
#             cat002.jpg
#             ...
#     validation/
#         dogs/
#             dog001.jpg
#             dog002.jpg
#             ...
#         cats/
#             cat001.jpg
#             cat002.jpg
#             ...

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.preprocessing import image
from keras.models import Model

from utility import show_train_history

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 20
batch_size = 16
train_history = None 

def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save('bottleneck_features_train.npy',bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save('bottleneck_features_validation.npy',bottleneck_features_validation)

def train_top_model():
    train_data = np.load('bottleneck_features_train.npy')
    train_labels = np.array([0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = np.array([0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    
    # or use ordinary vgg16's relu layer instead of dropout here, 
    # #like https://github.com/anujshah1003/Transfer-Learning-in-keras---custom-data/blob/master/transfer_learning_vgg16_custom_data.py
    model.add(Dropout(0.5)) 
    
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    global train_history
    train_history = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    show_train_history(train_history,'acc','val_acc')     
    show_train_history(train_history,'loss','val_loss')              
         
    model.save_weights(top_model_weights_path)

def predict():
    # from keras.applications.vgg16 import preprocess_input, decode_predictions

    # copy from classifier_from_little_data_script_3
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))
    top_model.load_weights(top_model_weights_path)
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    # alternative way: https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069#gistcomment-2578661
    # use base_modeo to predict once as the input of 2nd predict (then use top_model to predict final) 

    # 10.jpg # cat
    # 132.jpg # dog
    img = image.load_img('test_dog_132.jpg', target_size=(img_width, img_height))
    x = image.img_to_array(img)  # 150, 150, 3 (ndarray)
    x = np.expand_dims(x, axis=0) # 1, 150, 150, 3

    # probability, input samples [0], only 1 input, so output [result1], result is [0]
    # [0] for cat, [1] for dog. output is single neuron since Dense(1
    probs = model.predict(x) 
    print("get prediction:{}".format(probs))

def main():
    # save_bottleneck_features()
    train_top_model()
    # predict()
    print("done")    

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")


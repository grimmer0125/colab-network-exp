# @saulthu https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975#gistcomment-2068023
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense

from utility import show_train_history

# path to the model weights files.
weights_path = 'fine_tuned_weights.h5'
top_model_weights_path = 'bottleneck_fc_model-20epoch.h5'
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
test_data_dir = 'data/test'
nb_train_samples = 2000 #160 #2000
nb_validation_samples = 800 # 800
nb_test_samples = 800
epochs = 20 # 20 #50
batch_size = 16
train_history = None 

def define_model():
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
    print('base Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))    
    return model

def fine_tune_model(): 
    # build the VGG16 network
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
    print('base Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    # model.add(top_model)
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    # set the first 15 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:15]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    model.summary()

    # fine-tune the model
    global train_history
    train_history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        verbose=2)

    show_train_history(train_history,'acc','val_acc')     
    show_train_history(train_history,'loss','val_loss')              
                 
    # TODO 1. draw the acc graph 2. save the weight 3. use colab to train             
    model.save_weights(weights_path)

def evaluate_model():
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
        
    model = define_model()
    model.load_weights(weights_path)
    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                metrics=['accuracy'])    
    scores = model.evaluate_generator(generator, nb_test_samples // batch_size)
    print(scores[1]) # 0.32. scores: [0.32, 0.935] loss, acc

def main():
    print("start") 
    # fine_tune_model()
    evaluate_model()
    print("done") 

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")


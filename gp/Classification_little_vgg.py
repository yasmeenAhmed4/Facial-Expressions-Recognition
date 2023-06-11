from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

num_classes = 5
img_rows,img_cols = 48,48

batch_size = 32

train_data_dir = 'C:/Users/MSM/Desktop/gp/train'
validation_data_dir = 'C:/Users/MSM/Desktop/gp/validation'

train_datagen = ImageDataGenerator(
    #rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided
					rescale=1./255,
    #Int. Degree range for random rotations.
					rotation_range=30,
    #Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)                
					shear_range=0.3,
    #Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range]                
					zoom_range=0.3,
    #It may happen that the object may not always be in the center of the image
    #If the value is a float number, that would indicate the percentage of width or height of the image to shift.
    # Otherwise, if it is an integer value then simply the width or height are shifted by those many pixel values.
					width_shift_range=0.4,
					height_shift_range=0.4,
    #Boolean. Randomly flip inputs horizontally.                
					horizontal_flip=True,
    #When the image is rotated, some pixels will move outside the image and leave an empty area that needs to be filled in.
    #  You can fill this in different ways like a constant value or nearest pixel values, etc. This is specified in the fill_mode argument and 
    # the default value is “nearest” which simply replaces the empty area with the nearest pixel values.
					fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

#method allows you to read the images directly from the directory and augment them 
# while the neural network model is learning on the training data.
train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(img_rows,img_cols),
					batch_size=batch_size,
# Set “binary” if you have only two classes to predict, if not set to“categorical”,
#  in case if you’re developing an Autoencoder system,
#  both input and the output would probably be the same image, for this case set to “input”.                    
					class_mode='categorical',
#Set True if you want to shuffle خلط the order of the image that is being yielded, else set False.                    
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(img_rows,img_cols),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)

#There are two ways to build Keras models: sequential and functional.
#The sequential API allows you to create models layer-by-layer
#The functional API in Keras is an alternate way of creating models that offers a lot more flexibility, 
# including creating more complex models.
model = Sequential()

# Block-1

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
#standardize the inputs to a layer in a deep learning neural network.
#speed up the process of training
#It does so by applying a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
#the probability of setting each input to the layer to zero
model.add(Dropout(0.2))

# Block-2 

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-3

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-4 

model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-5

model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-6

model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-7

model.add(Dense(num_classes,kernel_initializer='he_normal'))
# is a mathematical function that converts a vector of numbers into a vector of probabilities,
#  where the probabilities of each value are proportional to the relative scale of each value in the vector.
model.add(Activation('softmax'))

print(model.summary())

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint(r'C:\Users\MSM\Desktop\gp\Emotion_little_vgg.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

nb_train_samples = 24176
nb_validation_samples = 3006
epochs=25

history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)
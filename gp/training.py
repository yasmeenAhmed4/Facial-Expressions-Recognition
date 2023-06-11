#import tensorflow as tf
#from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

num_classes = 5
img_rows,img_cols = 48,48
batch_size = 16 #32 #max num of input samples that use in one iteration in train

train_d_dir = r'C:\Users\hayay\Downloads\gp\keras\Facial-Expressions-Recognition-master\Facial-Expressions-Recognition-master\train'
valid_d_dir = r'C:\Users\hayay\Downloads\gp\keras\Facial-Expressions-Recognition-master\Facial-Expressions-Recognition-master\valid'
#do processing to make easy to train images  //Shear Intensity (Shear angle in counter-clockwise direction in degrees) get anothers imgs diffrent
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=30,
shear_range=0.3,zoom_range=0.3, #shear to increase images so that computers can see how humans see things from different angles
width_shift_range=0.4,height_shift_range=0.4,
horizontal_flip=True,fill_mode='nearest') #get max num of  different imgs

validation_datagen = ImageDataGenerator(rescale=1./255) #divide on max num of pixels

#any img came from train_d_dir which i make on it increase imgs then do this processing on it
train_generator = train_datagen.flow_from_directory(train_d_dir, #path
color_mode='grayscale',target_size=(img_rows,img_cols),
batch_size=batch_size,class_mode='categorical',shuffle=True) #catecorical means divide into categories like happy sad ,... shuffle mean not order organized 1st img can be happy when train 2nd can be sab and so on ..look varity of data at once

validation_generator = validation_datagen.flow_from_directory(valid_d_dir,
color_mode='grayscale',target_size=(img_rows,img_cols),batch_size=batch_size,
class_mode='categorical',shuffle=True)


model = Sequential()
##################

# Block-1
# 32 the num of filters or objects to detect ((3,3)determines the dimensions of the kernel.2 integers, specifying the height and width of the 2D convolution),padding spatial dimensions such that the output volume size matches the input volume size if it same,k_intit is  initialize all the values in the Conv2D class by 0 before actually training the model,
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))#1for gray scale
# thke the output of the layer input to yhe next layer ,if linear or non linear function specifies the name of the activation function you want to apply after performing the convolution.and it returns 0 if it receives any negative input ,tend to converge cost to zero faster and produce more accurate results. Different to other activation functions, ELU has a extra alpha constant which should be positive number.
model.add(Activation('elu')) # if positive it will be the same but it handle negitive value by equation output is close to 0 centrelized 
#improve the speed of training n network, reduce num of training , dont to get more deeper
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) #get max important features
#to  reduce overfitting and improve generalization error in deep neural networks
model.add(Dropout(0.4)) #0.3

# Block-2 
#64 n of nerons became 64
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))#0.3

# Block-3

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))#0.3

# Block-4 
#make con2d and max pool until reach 256 neron 
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))#0.3


# Block-5
#make feature map to one column (vector) data input layer to ann
model.add(Flatten())
#dense layer is a fully connected layer, meaning all the neurons in a layer are connected to those in the next layer

model.add(Dense(64,kernel_regularizer=keras.regularizers.L2(0.001),kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-6

model.add(Dense(64,kernel_regularizer=keras.regularizers.L2(0.001),kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-7...


#softmax used as the activation for the last layer of a classification network because the result could be interpreted as a probability distribution.
model.add(Dense(num_classes,kernel_regularizer=keras.regularizers.L2(0.001),kernel_initializer='he_normal')) #Regularizer to apply a penalty on the layer's kernel
model.add(Activation('softmax')) #for the output layer in multicclass classification problem if i have more 2 output categories probability make a fomla and take the highest probility as a final output

print(model.summary())


from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
 
 #save model has the best accuracy
checkpoint = ModelCheckpoint(r'C:\Users\hayay\Downloads\gp\keras\Facial-Expressions-Recognition-master\Facial-Expressions-Recognition-master\Emotion_little_vgg.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

#if not increase the accuracy stop training
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )
#if not improving reduce learning rate ,learn slowly 
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)



callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

nb_train_samples = 84878 #28821 #28,721 #24176 #24944
nb_validation_samples = 16041#17356 #7066 #28,721 #3006 #
epochs=25 #25

history=model.fit_generator(
               train_generator,
              steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
              validation_steps=nb_validation_samples//batch_size)




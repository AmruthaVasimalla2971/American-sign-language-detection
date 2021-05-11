from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers 
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import h5py


classifier = Sequential()
classifier.add(Convolution2D(128,(3,3),padding='same',activation='relu',input_shape = (28,28,3)))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(64,(3,3),strides=1,padding='same',activation='relu'))
classifier.add(Convolution2D(64,(3,3),strides=1,padding='same',activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(25,(3,3),strides=1,padding='same',activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(512,activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(26,activation='softmax'))
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/training_set',target_size=(28, 28),batch_size=32,class_mode='categorical')
test_set = test_datagen.flow_from_directory('data/test_set',target_size=(28, 28),batch_size=32,class_mode='categorical')

model = classifier.fit_generator(training_set,steps_per_epoch=800,epochs=10,validation_data = test_set,validation_steps = 6500)
classifier.summary()
classifier.save('Trained_model2.h5')


plt.plot(model.history['acc'])
plt.plot(model.history['val_acc'])
plt.title('model-2 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()

plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model-2 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()
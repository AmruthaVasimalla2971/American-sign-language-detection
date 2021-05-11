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
classifier.add(Convolution2D(32,(3,3),input_shape = (64 ,64 ,3), activation ='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Convolution2D(64,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(256,activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(26,activation='softmax'))
classifier.compile(optimizer = optimizers.SGD(lr = 0.01),loss = 'categorical_crossentropy',metrics = ['acc'])
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('data/training_set',target_size=(64, 64),batch_size=32,class_mode='categorical')
test_set = test_datagen.flow_from_directory('data/test_set',target_size=(64, 64),batch_size=32,class_mode='categorical')


model = classifier.fit_generator(training_set,steps_per_epoch=100,epochs=10,validation_data = test_set,validation_steps = 600)

classifier.save('Trained_model1.h5')

Accuracy
plt.plot(model.history['acc'])
plt.plot(model.history['val_acc'])
plt.title('model-1 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()

#Loss
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model-1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()






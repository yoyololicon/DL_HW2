from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Activation, Flatten, Dense
import os

scale_height = scale_width = 64
num_class = 11
batch_size = 128

data_dir = '/media/ycy/86A4D88BA4D87F5D/DataSet/Food-11'
train_dir = os.path.join(data_dir, 'training')
val_dir = os.path.join(data_dir, 'validation')
eva_dir = os.path.join(data_dir, 'evaluation')

train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

model = Sequential()
model.add(Conv2D(6, (5, 5), input_shape=(scale_height, scale_width, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(24, (5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(512, activation='sigmoid'))
model.add(Dense(num_class, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(scale_height, scale_width), batch_size=batch_size)
test_generator = test_datagen.flow_from_directory(eva_dir, target_size=(scale_height, scale_width), batch_size=batch_size)

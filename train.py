# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import os

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sz = 128

# Step 1 - Building the CNN
# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding fully connected layers with dropout
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=27, activation='softmax'))  # softmax for more than 2

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # categorical_crossentropy for more than 2

# Model Summary
classifier.summary()

# Step 2 - Preparing the train/test data and training the model
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(sz, sz),
                                                 batch_size=10,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(sz, sz),
                                            batch_size=10,
                                            color_mode='grayscale',
                                            class_mode='categorical')

# Training the model
classifier.fit(
    training_set,
    epochs=10,
    validation_data=test_set)

# Saving the model
model_json = classifier.to_json()
with open("model/model-bw.json", "w") as json_file:
    json_file.write(model_json)

print('Model Saved')

classifier.save_weights('model/model-bw.h5')
print('Weights saved')

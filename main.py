import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Preprocessing the Training set using ImageDataGenerator for data augmentation and normalization
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Loading and augmenting the training data from the 'PetImages' directory
training_set = train_datagen.flow_from_directory('PetImages',
                                                 target_size=(64, 64),  # Resizing images to 64x64 pixels
                                                 batch_size=32,
                                                 class_mode='binary')  # Binary classification mode

# Building the Convolutional Neural Network (CNN) model
cnn = tf.keras.models.Sequential()

# Step 1 - Adding the first convolutional layer
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(BatchNormalization())  # Batch normalization to improve training stability
cnn.add(MaxPooling2D(pool_size=2, strides=2))  # Max pooling to downsample the feature maps

# Adding a second convolutional layer
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=2, strides=2))

# Step 3 - Flattening the output from the convolutional layers to feed it into the fully connected layers
cnn.add(Flatten())

# Step 4 - Adding a fully connected hidden layer
cnn.add(Dense(units=128, activation='relu'))
cnn.add(Dropout(0.5))  # Dropout layer to reduce overfitting by randomly disabling neurons during training

# Step 5 - Output Layer with sigmoid activation for binary classification
cnn.add(Dense(units=1, activation='sigmoid'))

# Print the model summary to show the layers and the number of parameters
cnn.summary()

# Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN on the Training set
#cnn.fit(x=training_set, epochs=32)

# Saving the trained model to a file named 'model3.h5'
#cnn.save('model3.h5')

# The training process outputs loss and accuracy metrics for each epoch, like this:
# Epoch 32/32
# 314/314 [==============================] - 38s 121ms/step - loss: 0.2566 - accuracy: 0.8878

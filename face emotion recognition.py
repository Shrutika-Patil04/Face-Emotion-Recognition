import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

# Load the FER-2013 dataset (ensure this file is in your Colab environment)
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/fer2013.csv')

# Check the structure of the dataset
print(data.info())
print(data.head())

print(data.columns)

# Check the number of pixel values in each row
pixel_lengths = data['pixels'].apply(lambda x: len(x.split()))

# Filter out rows that don't have 2304 pixels (48x48)
print(f'Number of rows with incorrect pixel counts: {sum(pixel_lengths != 2304)}')

# Display rows with incorrect pixel counts
invalid_rows = data[pixel_lengths != 2304]
print(invalid_rows)

def preprocess_data(data):
    # Convert the pixel column into an array of integers
    X = np.array([np.reshape(np.array(pixels.split(), dtype='float32'), (48, 48, 1)) for pixels in data['pixels'].values])

    # Normalize the pixel values (between 0 and 1)
    X = X / 255.0

    # Convert emotion labels to categorical (one-hot encoding)
    y = to_categorical(data['emotion'], num_classes=7)

    return X, y

# Remove rows that do not have 2304 pixel values
data_cleaned = data[pixel_lengths == 2304].copy()

# Preprocess the cleaned dataset
X, y = preprocess_data(data_cleaned)

# Check the shape again
print(f'Input data shape: {X.shape}')
print(f'Labels shape: {y.shape}')

#from sklearn.model_selection import train_test_split

# Split the data into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the training and validation data
print(f'Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}')
print(f'Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Fit the data augmentation on the training data
#datagen.fit(X_train)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the CNN model
model = Sequential()

# Convolutional Layer 1
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 2
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 3
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output from the convolutional layers
model.add(Flatten())

# Dense (fully connected) layer
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Dropout layer to prevent overfitting

# Output layer (7 emotion classes)
model.add(Dense(7, activation='softmax'))

# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                    validation_data=(X_val, y_val),
                    epochs=30)

# Evaluate the model performance on the validation set
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f'Validation Accuracy: {val_acc * 100:.2f}%')

import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

from google.colab import files

# Upload an image file
uploaded = files.upload()

# Get the filename of the uploaded image
Face_1 = list(uploaded.keys())[0]
print(f'Uploaded file: {Face_1}')

import cv2
import numpy as np

# Function to preprocess the input image
def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to 48x48 pixels
    img_resized = cv2.resize(img_gray, (48, 48))

    # Normalize the pixel values
    img_normalized = img_resized / 255.0

    # Reshape to match model input shape
    img_reshaped = img_normalized.reshape(1, 48, 48, 1)  # (1, 48, 48, 1)

    return img_reshaped

# Preprocess the uploaded image
input_image = preprocess_image(Face_1)

# Make predictions
predictions = model.predict(input_image)

# Get the predicted emotion label
predicted_emotion = np.argmax(predictions)

# Mapping from integer to emotion label
emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
predicted_label = emotion_mapping[predicted_emotion]
print(f'Predicted Emotion: {predicted_label}')

# Display the input image with the predicted label
import matplotlib.pyplot as plt

# Load the original image for display
img_display = cv2.imread(Face_1)
img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying

plt.imshow(img_display)
plt.title(f'Predicted Emotion: {predicted_label}')
plt.axis('off')  # Hide axis
plt.show()

# Step 1: Upload the Image
from google.colab import files
uploaded = files.upload()
filename = list(uploaded.keys())[0]

# Step 2: Preprocess the Image
import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (48, 48))
    img_normalized = img_resized / 255.0
    img_reshaped = img_normalized.reshape(1, 48, 48, 1)
    return img_reshaped

input_image = preprocess_image(filename)

# Step 3: Make Predictions
predictions = model.predict(input_image)
predicted_emotion = np.argmax(predictions)
emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
predicted_label = emotion_mapping[predicted_emotion]

# Step 4: Display the Result
import matplotlib.pyplot as plt

img_display = cv2.imread(filename)
img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

plt.imshow(img_display)
plt.title(f'Predicted Emotion: {predicted_label}')
plt.axis('off')
plt.show()


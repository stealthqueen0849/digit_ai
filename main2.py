import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

train_data = pd.read_csv('digit-recognizer/train.csv')

X_train_full = train_data.iloc[:, 1:].values  # Pixel data
y_train_full = train_data.iloc[:, 0].values   # Labels

X_train_full = X_train_full / 255.0  # Normalize pixel values
X_train_full = X_train_full.reshape(-1, 28, 28)  # Reshape to (num_samples, 28, 28)

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3, validation_data=(X_val, y_val))

model.save('handwritten_csv.model.keras')

model = tf.keras.models.load_model('handwritten.model.keras')

loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

test_data = pd.read_csv('digit-recognizer/test.csv')

# Normalize and reshape the test data
X_test = test_data.values / 255.0  # Normalize pixel values
X_test = X_test.reshape(-1, 28, 28)  # Reshape to (num_samples, 28, 28)

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))  # Resize to match input shape
        img = tf.keras.utils.normalize(img, axis=1)  # Normalize the image
        img = np.array(img).reshape(1, 28, 28)  # Reshape for prediction
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    
    finally:
        image_number += 1

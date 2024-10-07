# Training and saving the model
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Function to extract MFCC features from audio files
def extract_mfcc(audio_file, max_pad_len=174):
    audio, sr = librosa.load(audio_file, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    # Ensure the fixed length
    if mfccs.shape[1] > max_pad_len:
        mfccs = mfccs[:, :max_pad_len]
    else:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    return mfccs

# Function to load and process the audio data
def load_data(data_folder):
    mfccs = []
    labels = []

    for folder in os.listdir(data_folder):
        label = 'Spark' if folder == 'Spark' else 'Not_Spark'
        folder_path = os.path.join(data_folder, folder)

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            mfcc = extract_mfcc(file_path)
            mfccs.append(mfcc)
            labels.append(label)

    return np.array(mfccs), np.array(labels)

# Load and preprocess the data
data_folder = "C://Users//srika//Conceptual Project//Sounds we need to classify"
X, y = load_data(data_folder)

# Convert labels to one-hot encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Reshape data for CNN input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # Two classes: 'Spark' and 'Not_Spark'

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save('spark_detection_model.h5')
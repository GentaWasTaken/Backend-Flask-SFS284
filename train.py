import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (128, 128)
TRAIN_DIR = 'train/train'

def load_data():
    images = []
    labels = []
    
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory '{TRAIN_DIR}' not found.")
    
    class_dirs = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d)) and d.isdigit()]
    
    class_dirs = sorted(class_dirs, key=int)
    
    if len(class_dirs) != 20:
        raise ValueError(f"Expected exactly 20 numeric subdirectories (but found {len(class_dirs)}) in the '{TRAIN_DIR}' directory.")

    for class_id, class_dir in enumerate(class_dirs):
        class_path = os.path.join(TRAIN_DIR, class_dir)
        
        flower_images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        if not flower_images:
            print(f"No valid images found in class directory {class_dir}. Skipping.")
            continue

        for img_name in flower_images:
            img_path = os.path.join(class_path, img_name)

            try:
                img = Image.open(img_path)
                img = img.resize(IMAGE_SIZE)
                img = image.img_to_array(img)
                img = img / 255.0
                images.append(img)
                labels.append(class_id)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

    return np.array(images), np.array(labels)

X, y = load_data()

if len(X) == 0:
    raise ValueError("No images found in the dataset. Check the image directory structure.")

y = to_categorical(y, num_classes=20)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(20, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

model.save('flower_classifier_model.h5')

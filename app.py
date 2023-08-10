import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load EfficientNetB0 with pre-trained weights
base_model = EfficientNetB0(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Load saved weights of the trained mask detection model
model.load_weights('research/mask_detection_model.h5')

# Open a webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame for inference
    resized_frame = cv2.resize(frame, (224, 224))
    input_data = np.expand_dims(resized_frame, axis=0)
    input_data = tf.keras.applications.efficientnet.preprocess_input(input_data)

    # Perform inference
    predictions = model.predict(input_data)
    class_id = np.argmax(predictions)
    confidence = predictions[0, class_id]

    label = 'With Mask' if class_id == 0 else 'Without Mask'
    color = (0, 255, 0) if class_id == 0 else (0, 0, 255)

    cv2.putText(frame, f'{label} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Mask Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

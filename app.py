from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_hub as hub

app = Flask(__name__)
camera = cv2.VideoCapture(0) 
mask_detection_model = load_model('mask_detect_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})

def generate_frames():
    while True:
        success, frame = camera.read()
        
        if not success:
            break
        else:
            # For Face Detection
            frontal_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faces = frontal_face.detectMultiScale(frame, 1.1, 7)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for (x, y, w ,h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

            # Mask Detection
            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                face_roi_resized = cv2.resize(face_roi, (224, 224))
                face_roi_normalized = face_roi_resized / 255.0
                face_roi_input = tf.expand_dims(face_roi_normalized, axis=0)

                # Convert grayscale image to color (3 channels)
                face_roi_color = cv2.cvtColor(face_roi_resized, cv2.COLOR_GRAY2BGR)
    
                # Ensure the input shape matches (None, 224, 224, 3)
                face_roi_input = tf.expand_dims(face_roi_color, axis=0)

                prediction = mask_detection_model.predict(face_roi_input)
                mask_probability = prediction[0][0]  # Probability of wearing a mask

                label = "Mask" if mask_probability > 0.5 else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        
        yield(b'--frame\r\n'
                    b'content-type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

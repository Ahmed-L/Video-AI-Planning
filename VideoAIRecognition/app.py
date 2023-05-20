from flask import Flask, Response
import cv2
import dlib
import numpy as np
import tensorflow as tf
app = Flask(__name__)
# Path to the FaceNet model file and weights
facenet_model_path = 'path/to/your/facenet_model.pb'
# Load the pre-trained FaceNet model and other necessary configurations
face_net_model = tf.keras.models.load_model(facenet_model_path)
# Initialize the dlib face detector
face_detector = dlib.get_frontal_face_detector()
# Function to process frames from the video stream
def process_frames():
# Open the RTSP stream using OpenCV
stream = cv2.VideoCapture('rtsp://your_rtsp_stream_url')
while True:
ret, frame = stream.read()
if not ret:
break
# Perform face detection using dlib
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_detector(gray_frame)
# Extract faces and send for recognition
for rect in faces:
x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
face_img = frame[y:y+h, x:x+w]
BUET Video Analytics planning 5
# Preprocess the face image (resize, normalize, etc.) as required by the FaceNet model
# Perform face recognition using the FaceNet model
face_embedding = face_net_model.predict(face_img)
# Perform further processing or store the recognized face embedding
# Display the processed frame (optional)
cv2.imshow('Processed Frame', frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
break
# Save unrecognized embeddings to database
for emb, bb in zip(embeddings, bounding_boxes):
if bb not in recognized_bounding_boxes:
# Save embedding to database with a flag indicating it's unrecognized
db.session.add(FaceEmbedding(emb=emb, recognized=False))
db.session.commit()
stream.release()
cv2.destroyAllWindows()
# Route for video streaming
@app.route('/video_feed')
def video_feed():
return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
# Run the Flask application
if __name__ == '__main__':
app.run(debug=True)

from PIL import Image
import numpy as np

import cv2
from flask import app, Response, Blueprint, jsonify
from mtcnn import MTCNN
from inceptionv3 import preprocess, test_inference, getEmbeddingFromImage, getTopKSimilarities
import logging
import os


video = Blueprint('api', __name__)

# Define the text and its properties
text = 'Hello, OpenCV!'
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
color = (0, 255, 0)  # BGR color tuple
thickness = 2

mtcnn = MTCNN()

def process_frames(verify=False):
    # Open the RTSP stream using OpenCV
    # stream = cv2.VideoCapture('rtsp://your_rtsp_stream_url')
    cap = cv2.VideoCapture(0)
    while True:
        # ret, frame = stream.read()
        ret, frame = cap.read()
        if not ret:
            print("no cam detected")
            break

        # Perform face detection using MTCNN
        faces = mtcnn.detect_faces(frame)

        # Extract faces and send for recognition
        for face in faces:
            x, y, w, h = face['box']
            face_img = frame[y:y + 2 * h, x:x + 2 * w]

            # Preprocess the face image (resize, normalize, etc.) # Will be performed inside the v3 module by
            # getEmbeddings(), Perform face recognition using the inceptionV3
            if verify:
                pil_image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                face_tensor = getEmbeddingFromImage(pil_image)
                getTopKSimilarities(top_k=3, face_embedding=face_tensor)
                # face_tensor = test_inference(face_img, top_k=3)

            # face_embedding = face_net_model.predict(np.expand_dims(face_img, axis=0))

            # Perform further processing or store the recognized face embedding

            # Display the processed frame (optional)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, text, (50, 50), font, font_scale, color, thickness)

        # Display the processed frame (optional)
        cv2.imshow('Processed Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # stream.release()
    cap.release()
    cv2.destroyAllWindows()


process_frames(verify=True)


# frame capture test
# def generate_frames():
#     cap = cv2.VideoCapture(1)
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Convert the frame to JPEG format
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
#
#         # Yield the frame as response
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#
#     cap.release()


@video.route('/test')
def test():
    return jsonify({'message': 'Hello, World!'})


# Route for video streaming
@video.route('/video_feed')
def video_feed():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

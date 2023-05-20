from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
from mtcnn import MTCNN
from inceptionv3 import preprocess, test_inference, getEmbeddingFromImage, getTopKSimilarities

mtcnn = MTCNN()


def process_frames(verify=False):
    # Open the RTSP stream using OpenCV
    # stream = cv2.VideoCapture('rtsp://your_rtsp_stream_url')
    cap = cv2.VideoCapture(1)
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
            face_img = frame[y:y + 2*h, x:x + 2*w]

            # Preprocess the face image (resize, normalize, etc.) # Will be performed inside the v3 module by getEmbeddings()
            # Perform face recognition using the inceptionV3
            if(verify):
                pil_image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                face_tensor = getEmbeddingFromImage(pil_image)
                getTopKSimilarities(top_k=3, face_embedding=face_tensor)
                # face_tensor = test_inference(face_img, top_k=3)

            # face_embedding = face_net_model.predict(np.expand_dims(face_img, axis=0))

            # Perform further processing or store the recognized face embedding

            # Display the processed frame (optional)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the processed frame (optional)
        cv2.imshow('Processed Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # stream.release()
    cap.release()
    cv2.destroyAllWindows()


# def detectFaces():

process_frames(verify=True)

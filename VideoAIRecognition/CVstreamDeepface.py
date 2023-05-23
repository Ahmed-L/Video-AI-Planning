from deepface import DeepFace
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

ret = ''
similarity = 0
identity = ''
# Define the text properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
color = (0, 255, 0)  # BGR color tuple
thickness = 2

model = tf.keras.models.load_model('../ArcFaceModel/arcface_model')


def process_frames(verify=False):
    # Open the RTSP stream using OpenCV
    # stream = cv2.VideoCapture('rtsp://your_rtsp_stream_url')
    text = ''
    cap = cv2.VideoCapture(0)
    while True:
        # ret, frame = stream.read()
        ret, frame = cap.read()
        if not ret:
            print("no cam detected")
            break

        # Perform face detection using -ssd
        # faces = mtcnn.detect_faces(frame)

        # now using deepface's ssd to extract faces
        frame_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret = DeepFace.find(frame_array, '../images', 'ArcFace', 'cosine', enforce_detection=False,
                            detector_backend='opencv', align=True, prog_bar=False, normalization='ArcFace')

        similarity = 1
        if ret is not None:
            try:
                similarity = ret.iloc[0, 1]
                identity = str(ret.iloc[0, 0])
                identity = identity.split("/")[-1]
                identity = identity.split(".")[0]
            except:
                pass

        text = ''

        if similarity < 0.55:
            text = 'Name: ' + identity + ' similarity: ' + str(similarity)
        else:
            text = 'Unrecognized, dist: ' + str(similarity)

        # text = 'Name: ' + identity + ' similarity: ' + str(similarity)
        cv2.putText(frame, text, (50, 50), font, font_scale, color, thickness)

        # Display the processed frame (optional)
        cv2.imshow('Processed Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # stream.release()
    cap.release()
    cv2.destroyAllWindows()

    # print(ret.shape)
    # print(ret.head())
    # print(ret[0])


process_frames(verify=False)

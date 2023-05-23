from deepface import DeepFace


DeepFace.stream('../images', model_name='Facenet512', detector_backend='ssd', enable_face_analysis=False, time_threshold=2, frame_threshold=2)
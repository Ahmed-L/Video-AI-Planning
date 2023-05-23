from deepface import DeepFace

# Load the pre-trained model for face recognition
# model_name = "ArcFace"
# model = DeepFace.build_model(model_name)
#
# # Define the paths to the images
# image_path1 = '../images/Oliver Goodwill.jpg'
# image_path2 = '../images/faisal.jpg'
#
# # Perform face recognition
# result = DeepFace.verify(image_path1, image_path2, model_name=model_name)
#
# # Print the result
# print("Is the same person?", result["verified"])
# print("Similarity score:", result["distance"])
DeepFace.stream(db_path='../images', model_name='ArcFace', detector_backend='opencv', distance_metric='cosine', enable_face_analysis=False, time_threshold=2, frame_threshold=2)


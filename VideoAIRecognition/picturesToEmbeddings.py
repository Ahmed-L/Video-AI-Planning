import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pickle
import cv2 as cv2

from mtcnn import MTCNN

from inceptionv3 import getEmbeddingFromImage
import os
from PIL import Image

mtcnn = MTCNN()

images = []
image_list = []
embeddingsDict = {}

image_directory = '../images'  # Replace with your image directory


# Iterate over all files in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg"):  # Replace with the desired image file extension
        image_path = os.path.join(image_directory, filename)
        image_list.append(image_path)
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        face = mtcnn.detect_faces(img)
        embeddingsDict[filename] = getEmbeddingFromImage(face, False)


# Save the `embeddingsDict` to a file
output_file = 'embeddings_dict.pkl'  # Specify the desired output file name
with open(output_file, 'wb') as f:
    pickle.dump(embeddingsDict, f)

print("Embeddings dictionary saved to", output_file)

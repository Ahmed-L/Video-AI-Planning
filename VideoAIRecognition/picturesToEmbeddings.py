import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pickle
from inceptionv3 import getEmbeddingFromImage
import os

images = []
image_list = []
embeddingsDict = {}

image_directory = '../images'  # Replace with your image directory


# Iterate over all files in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg"):  # Replace with the desired image file extension
        image_path = os.path.join(image_directory, filename)
        image_list.append(image_path)
        embeddingsDict[filename] = getEmbeddingFromImage(image_path)


# Save the `embeddingsDict` to a file
output_file = 'embeddings_dict.pkl'  # Specify the desired output file name
with open(output_file, 'wb') as f:
    pickle.dump(embeddingsDict, f)

print("Embeddings dictionary saved to", output_file)

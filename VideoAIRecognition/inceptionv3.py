import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle

# Load the pretrained FaceNet model
model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)  # No need to re-download

# Load the pretrained Inception_v3 model
# model = models.inception_v3(weights=None, init_weights=True)  # weights=None == pretrained=False
# path = "../inceptionv3Model/inception_v3_google-0cc3c7bd.pth"
# model.load_state_dict(torch.load(path))
model.eval()

# Define a transformation to preprocess the input images
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load a sample image for testing
def getEmbeddingFromImage(image, convert=False):
    if convert:
        image = Image.open(image).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # If you have a GPU, move the input tensor to the GPU
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Perform face recognition
    with torch.no_grad():
        # Forward pass through the network
        output = model(input_batch)

    # Process the output to get the face embeddings
    face_embedding = torch.nn.functional.normalize(output, p=2, dim=1).squeeze()
    return face_embedding


# Compare the face embeddings with a database of known faces
# Load the `embeddingsDict` from the saved file
database = {}
input_file = 'embeddings_dict.pkl'  # Specify the path to the saved file
with open(input_file, 'rb') as f:
    database = pickle.load(f)


# Function to compute the similarity between the face embedding and the known faces in the database
def getTopKSimilarities(top_k=3, face_embedding=None):
    similarities = {}
    for name, embedding in database.items():
        similarity = torch.cosine_similarity(face_embedding, embedding, dim=0)
        similarities[name] = similarity.item()

    # Sort the similarities in descending order
    similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    # Print the top-k most similar faces
    top_k = top_k
    for name, similarity in similarities[:top_k]:
        print(f'{name}: {similarity:.4f}')
        return str(f'{name}: {similarity:.4f}')


def test_inference(image, top_k):
    face_embedding = getEmbeddingFromImage(image)
    getTopKSimilarities(top_k, face_embedding)


# test_inference('../source.jpg', top_k=5)
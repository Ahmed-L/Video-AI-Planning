import tensorflow as tf
import numpy as np
import cv2

# Load FaceNet model
model_dir = '../from_facenet_repo'
model_meta_file = model_dir + '../from_facenet_repo/model-20170512-110547.meta'
model_ckpt_file = model_dir + '../from_facenet_repo/model-20170512-110547.ckpt-250000.data-00000-of-00001'

# Create a TensorFlow session
sess = tf.compat.v1.Session()

# Import the graph definition from the meta file
saver = tf.compat.v1.train.import_meta_graph(model_meta_file)

# Restore the weights from the checkpoint file
saver.restore(sess, model_ckpt_file)

# Get the input and output tensors from the loaded graph
input_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('input:0')
output_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('embeddings:0')


# Function to pre-process input images
def preprocess_image(image):
    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image to the required input size of FaceNet model
    image = cv2.resize(image, (160, 160))

    # Normalize pixel values to the range of 0-1
    image = image / 255.0

    # Expand dimensions to create a batch of size 1
    image = np.expand_dims(image, axis=0)

    return image

# Function to generate embeddings using FaceNet model
def generate_embeddings(images):
    # Preprocess input images
    preprocessed_images = np.array([preprocess_image(image) for image in images])

    # Generate embeddings using the FaceNet model
    embeddings = model.predict(preprocessed_images)

    return embeddings

# Load sample images for face recognition
image1 = cv2.imread('path/to/your/image1.jpg')
image2 = cv2.imread('path/to/your/image2.jpg')

# Generate embeddings for the sample images
embeddings = generate_embeddings([image1, image2])

# Perform face recognition or comparison
# ...

# Example: Compute Euclidean distance between the embeddings
distance = np.linalg.norm(embeddings[0] - embeddings[1])
print("Euclidean distance:", distance)

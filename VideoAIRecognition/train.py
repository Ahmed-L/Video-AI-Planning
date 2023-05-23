import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Set the path to your training and validation data directories
train_data_dir = 'path/to/your/training/data'
validation_data_dir = 'path/to/your/validation/data'

# Set the batch size and number of workers for data loaders
batch_size = 32
num_workers = 4

# Set the number of classes in your dataset
num_classes = 10

# Set the number of epochs and learning rate
num_epochs = 10
learning_rate = 0.001

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create data loaders for training and validation sets
train_dataset = ImageFolder(train_data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

validation_dataset = ImageFolder(validation_data_dir, transform=transform)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Load the pretrained InceptionV3 model
model = inception_v3(weights=None, init_weights=True)  # weights=None == pretrained=False
path = "../inceptionv3Model/inception_v3_google-0cc3c7bd.pth"
model.load_state_dict(torch.load(path))

# Replace the last fully connected layer with a new one
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = correct / total * 100
    return accuracy


# Start training
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_accuracy = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_accuracy += calculate_accuracy(outputs, labels) * images.size(0)

    # Calculate average loss and accuracy over the training set
    train_loss = train_loss / len(train_dataset)
    train_accuracy = train_accuracy / len(train_dataset)

    # Evaluate on the validation set
    model.eval()
    validation_loss = 0.0
    validation_accuracy = 0.0

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            validation_loss += loss.item() * images.size(0)
            validation_accuracy += calculate_accuracy(outputs, labels) * images.size(0)

        # Calculate average loss and accuracy over the validation set
        validation_loss = validation_loss / len(validation_dataset)
        validation_accuracy = validation_accuracy / len(validation_dataset)

        # Print epoch statistics
        print(f'Epoch: {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}%')

    # Save the trained model
    torch.save(model.state_dict(), 'path/to/save/model.pth')

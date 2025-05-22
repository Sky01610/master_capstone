import torch
from torchvision import models, transforms
from PIL import Image
import os

# Set device
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # For Mac Silicon, fallback to CPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained VGG16 model
def load_model(model_path):
    """
    Load VGG16 model with pre-trained weights and modify for binary classification.
    """
    # Load VGG16 model pretrained on ImageNet
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Modify the classifier to fit binary classification (2 classes)
    num_features = model.classifier[6].in_features  # Input size of the original final FC layer
    model.classifier[6] = torch.nn.Linear(num_features, 2)  # Replace with new linear layer for 2 output classes

    # Load the model weights from file
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Move the model to the correct device and set to evaluation mode
    return model.to(device).eval()


# Preprocessing for input images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Resize((224, 224)),  # Resize input image (VGG16 works with 224x224)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])


# Predict the label for a single image
def predict(model, image_path, class_names):
    """
    Predict the label of an image using the trained model.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)[0]  # Get probabilities

    predicted_label = class_names[predicted.item()]
    confidence = probabilities[predicted.item()] * 100

    return predicted_label, confidence.item()


# Test multiple images in a directory
def test_images(model, image_dir, class_names, expected_class):
    """
    Test and calculate the accuracy on multiple images in a directory.
    """
    correct_predictions = 0
    total_images = 0

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)

        if image_name.lower().endswith(("jpg", "jpeg", "png")):  # Ensure file is an image
            predicted_label, confidence = predict(model, image_path, class_names)

            # Update accuracy counters
            if str(expected_class) == str(predicted_label):
                correct_predictions += 1
            total_images += 1

            # Fixed this line to avoid formatting errors
            print(f"{predicted_label}, {confidence:.2f}%")  # Print prediction and confidence

    # Avoid division by zero and calculate accuracy
    accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy



# Example usage
if __name__ == "__main__":
    class_names = ["Cluster_Cell", "Single_Cell"]  # Define class mappings
    model_path = "model/classification/best_vgg16_model.pth"  # Path to the VGG16 model weights

    # Load the trained model
    model = load_model(model_path)

    # Test on a directory of images
    test_image_dir = "content/extract"  # Directory with test images
    test_images(model, test_image_dir, class_names, "Cluster_Cell")  # Specify expected class
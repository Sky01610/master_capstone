import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import KMeans  # We'll use KMeans instead of RandomForest since we don't have labels
import shutil  # To handle file movement
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.decomposition import PCA
import csv
from sklearn.decomposition import PCA

# Device configuration remains the same
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the VGG16 model for feature extraction
model = models.efficientnet_b0(pretrained=True)
#model  = models.vgg16(pretrained=True)
model.eval()

# Modify the model to remove the classification head (fully connected layers)
# Use only the convolutional layers for feature extraction
model = nn.Sequential(*list(model.features)).to(device)

# Image transformation remains the same
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_features_efficientnet(images):
    features = []
    for img in images:
        transformed_img = image_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model(transformed_img)
        features.append(feature.view(-1).cpu().numpy())
    return np.vstack(features)


def load_images(image_dir):
    images = []
    filenames = []  # Keep track of filenames for reference

    for file in os.listdir(image_dir):
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            file_path = os.path.join(image_dir, file)
            try:
                img = Image.open(file_path).convert('RGB')
                images.append(img)
                filenames.append(file)
            except Exception as e:
                print(f"Error loading image {file_path}: {e}")

    return images, filenames


# Configuration
image_dir = "categorized_cells/Single_cell"  # Update this to your folder path
n_clusters = 2  # Update this to your desired number of clusters

# Load data
images, filenames = load_images(image_dir)

# Extract features using EfficientNet
features = extract_features_efficientnet(images)

# Perform clustering using KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
predicted_clusters = kmeans.fit_predict(features)


# Visualization function
def plot_clusters(features, cluster_labels, save_path="result/clustering_result.png"):
    from sklearn.decomposition import PCA
    
    # Perform PCA on features
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # Define cluster colors: red for Cluster 0, blue for Cluster 1
    cluster_colors = ['red' if label == 0 else 'blue' for label in cluster_labels]

    # Plot scatter plot with updated colors
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1],
                c=cluster_colors, s=20, alpha=0.7)

    # Add legend
    #plt.legend(["Cluster 0", "Cluster 1"], loc="upper right")

    # Add plot titles and labels
    plt.title("PCA Clustering Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # Save result to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Plot saved as: {save_path}")


# Plot clustering visualization
plot_clusters(features, predicted_clusters)

# Print cluster assignments
for filename, cluster in zip(filenames, predicted_clusters):
    print(f"{filename}: Cluster {cluster}")

# Use KNN for confidence estimation
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
knn_model.fit(features, predicted_clusters)

# Directories for clustered images
cluster_dirs = [f"categorized_cells/Cluster_{i}" for i in range(n_clusters)]
for cluster_dir in cluster_dirs:
    os.makedirs(cluster_dir, exist_ok=True)

# Function to plot confidence distribution
def plot_confidence_distribution(confidences, save_path="result/confidence_distribution.png"):
    plt.figure(figsize=(10, 6))

    # Plot histogram
    plt.hist(confidences, bins=20, color='skyblue', edgecolor='black', alpha=0.7)

    # Add labels and title
    plt.title("Confidence Distribution", fontsize=16)
    plt.xlabel("Confidence", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)

    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confidence distribution plot saved as: {save_path}")


# Collect confidences while processing images:
confidences = []

for file, feature, cluster in zip(filenames, features, predicted_clusters):
    source_path = os.path.join(image_dir, file)

    # Predict cluster and calculate confidence
    probabilities = knn_model.predict_proba([feature])[0]  # List of probabilities per cluster
    confidence = probabilities[cluster]  # Confidence for the assigned cluster
    confidences.append(confidence)

    # Create a new filename with the confidence prepended
    confidence_str = f"{confidence:.3f}"  # Precision up to 3 decimal places
    new_file_name = f"{confidence_str}_{file}"
    target_dir = cluster_dirs[cluster]
    target_path = os.path.join(target_dir, new_file_name)

    try:
        # Copy the file to the new location with the modified name
        shutil.copy(source_path, target_path)
        print(f"Copied {file} to {target_dir} as {new_file_name}")
    except Exception as e:
        print(f"Error moving {file}: {e}")

# Plot the confidence distribution
plot_confidence_distribution(confidences)

def plot_overall_pca_distribution(features, save_path="result/pca_overall_distribution.png"):
    # Apply PCA to features
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(reduced_features[:, 0], shade=True, color="green", label="PCA Component 1")
    sns.kdeplot(reduced_features[:, 1], shade=True, color="orange", label="PCA Component 2")
    
    plt.title("Overall PCA Components Distribution")
    plt.xlabel("PCA Component Value")
    plt.ylabel("Density")
    plt.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_clusters_with_filenames(features, cluster_labels, filenames, save_path="result/clustering_result.png"):
    """
    Creates an interactive scatter plot for clusters with filenames displayed
    on hover using mplcursors.

    :param features: Features matrix (e.g., PCA-reduced features).
    :param cluster_labels: Cluster assignments for each data point.
    :param filenames: List of filenames corresponding to the data points.
    :param save_path: Path to save the plot image.
    """
    from sklearn.decomposition import PCA
    import mplcursors

    # Perform PCA to reduce features to 2 dimensions for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # Scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1],
                          c=cluster_labels, cmap='tab10', s=20, alpha=0.7)

    # Add hover interaction with mplcursors
    cursor = mplcursors.cursor(scatter, hover=True)

    # Annotate with filenames
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(filenames[sel.index])
        sel.annotation.set_backgroundcolor("white")

    # Add legend for clusters
    legend_labels = [f"Cluster {i}" for i in range(len(set(cluster_labels)))]
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, loc="upper right")

    # Add titles and labels
    plt.title("EfficientNet Features Clustering Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Interactive plot saved as: {save_path}")

def plot_combined_pca_distributions(features, cluster_labels, n_clusters, save_path="result/pca_combined_distribution.png"):
    # Perform PCA to reduce to 2 components
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    
    plt.figure(figsize=(14, 6))

    # PCA Component 1 Distribution
    plt.subplot(1, 2, 1)
    for cluster in range(n_clusters):
        cluster_data = reduced_features[cluster_labels == cluster, 0]  # PCA Component 1
        sns.kdeplot(cluster_data, shade=True, label=f"Cluster {cluster}")
    plt.title("PCA Component 1 Distribution")
    plt.xlabel("PCA Component 1")
    plt.ylabel("Density")
    plt.legend()

    # PCA Component 2 Distribution
    plt.subplot(1, 2, 2)
    for cluster in range(n_clusters):
        cluster_data = reduced_features[cluster_labels == cluster, 1]  # PCA Component 2
        sns.kdeplot(cluster_data, shade=True, label=f"Cluster {cluster}")
    plt.title("PCA Component 2 Distribution")
    plt.xlabel("PCA Component 2")
    plt.ylabel("Density")
    plt.legend()

    # Save and display the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

plot_combined_pca_distributions(features, predicted_clusters, n_clusters)

def plot_pca_with_kde(features, cluster_labels, n_clusters, save_path="result/pca_with_kde.png"):
    # Perform PCA to reduce to 2 components
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(15, 5))

    # Scatter plot and KDE for PCA Component 1 and 2
    for cluster in range(n_clusters):
        cluster_data = reduced_features[cluster_labels == cluster]
        
        # Scatter plot for current cluster
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=20, label=f"Cluster {cluster}", alpha=0.6)
        
        # KDE plot for current cluster
        sns.kdeplot(x=cluster_data[:, 0], y=cluster_data[:, 1], 
                    levels=10, color=f"C{cluster}", label=f"Cluster {cluster} Density",
                    alpha=0.5, linewidths=1)

    plt.title("PCA Components with KDE Distribution")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save and display the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# Usage
plot_pca_with_kde(features, predicted_clusters, n_clusters)


def plot_pca_distributions(features, predicted_clusters, n_clusters, save_dir="result"):
    from sklearn.decomposition import PCA
    from scipy import stats

    # Perform PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    colors = ['#FF4B4B', '#4B7BFF']  # Same colors as before

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(15, 5))

    # 1. PCA Component 1 Distribution
    plt.subplot(131)
    for i in range(n_clusters):  # Corrected from range(len(n_clusters)) to range(n_clusters)
        mask = np.array(predicted_clusters) == i
        component_1 = reduced_features[mask, 0]

        # Calculate KDE
        kde = stats.gaussian_kde(component_1)
        x_range = np.linspace(min(component_1), max(component_1), 200)

        plt.plot(x_range, kde(x_range), color=colors[i], label=f"Cluster {i}")
        plt.fill_between(x_range, kde(x_range), alpha=0.3, color=colors[i])

    plt.title("PCA Component 1 Distribution", fontsize=12)
    plt.xlabel("Component 1 Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    # 2. PCA Component 2 Distribution
    plt.subplot(132)
    for i in range(n_clusters):  # Corrected from range(len(n_clusters)) to range(n_clusters)
        mask = np.array(predicted_clusters) == i
        component_2 = reduced_features[mask, 1]

        # Calculate KDE
        kde = stats.gaussian_kde(component_2)
        x_range = np.linspace(min(component_2), max(component_2), 200)

        plt.plot(x_range, kde(x_range), color=colors[i], label=f"Cluster {i}")
        plt.fill_between(x_range, kde(x_range), alpha=0.3, color=colors[i])

    plt.title("PCA Component 2 Distribution", fontsize=12)
    plt.xlabel("Component 2 Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    # 3. 2D KDE plot
    plt.subplot(133)
    for i in range(n_clusters):  # Corrected from range(len(n_clusters)) to range(n_clusters)
        mask = np.array(predicted_clusters) == i
        x = reduced_features[mask, 0]
        y = reduced_features[mask, 1]

        # Calculate 2D KDE
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        # Create grid of points
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = stats.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)

        # Plot 2D KDE with contours
        plt.contour(xx, yy, f, colors=colors[i], alpha=0.6, levels=5)
        plt.contourf(xx, yy, f, colors=[colors[i]], alpha=0.3, levels=5)

        # Plot scatter points
        plt.scatter(x, y, c=colors[i], label=f"Cluster {i}",
                    alpha=0.4, s=20)

    plt.title("2D PCA Distribution", fontsize=12)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    save_path = os.path.join(save_dir, "pca_distributions.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"PCA distribution plots saved as: {save_path}")

    # Additional statistical information
    print("\nPCA Components Statistics:")
    for i in range(n_clusters):  # Corrected from range(len(n_clusters)) to range(n_clusters)
        mask = np.array(predicted_clusters) == i
        comp1 = reduced_features[mask, 0]
        comp2 = reduced_features[mask, 1]

        print(f"\nCluster {i}:")
        print(f"Component 1 - Mean: {np.mean(comp1):.3f}, Std: {np.std(comp1):.3f}")
        print(f"Component 2 - Mean: {np.mean(comp2):.3f}, Std: {np.std(comp2):.3f}")

# Usage
plot_pca_distributions(features, predicted_clusters, n_clusters, save_dir="result")

# Perform PCA on the extracted features
pca = PCA(n_components=2)
pca_features = pca.fit_transform(features)  # "features" is from your code

# Combine filenames with PCA results
pca_data = [(filenames[i], pca_features[i, 0], pca_features[i, 1]) for i in range(len(filenames))]

# Save to CSV file
output_file = "pca_features.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Filename", "PC1", "PC2"])  # Header
    writer.writerows(pca_data)

print(f"PCA features saved to {output_file}")

def plot_clusters_with_filenames(features, cluster_labels, filenames, save_path="result/clustering_result.png"):
    """
    Creates an interactive scatter plot for clusters with filenames displayed
    on hover using mplcursors.
    
    :param features: Features matrix (e.g., PCA-reduced features).
    :param cluster_labels: Cluster assignments for each data point.
    :param filenames: List of filenames corresponding to the data points.
    :param save_path: Path to save the plot image.
    """
    from sklearn.decomposition import PCA
    import mplcursors

    # Perform PCA to reduce features to 2 dimensions for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # Scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1],
                          c=cluster_labels, cmap='tab10', s=20, alpha=0.7)

    # Add hover interaction with mplcursors
    cursor = mplcursors.cursor(scatter, hover=True)

    # Annotate with filenames
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(filenames[sel.index])
        sel.annotation.set_backgroundcolor("white")

    # Add legend for clusters
    legend_labels = [f"Cluster {i}" for i in range(len(set(cluster_labels)))]
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, loc="upper right")

    # Add titles and labels
    plt.title("EfficientNet Features Clustering Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Interactive plot saved as: {save_path}")
    

plot_clusters_with_filenames(features, predicted_clusters, filenames)
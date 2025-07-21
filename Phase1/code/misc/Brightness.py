from sklearn.cluster import KMeans
import cv2
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

def generate_brightness_map(image_paths, num_clusters=16, image_save_path="BrightnessMap"):
    """
    Generates a brightness map for each image using K-means clustering on grayscale brightness values.
    
    Parameters:
        image_paths (list of str): List of image file paths.
        num_clusters (int): Number of clusters (brightness bins).
        save_prefix (str): Prefix for saved image filenames.
    
    Returns:
        list of np.array: List of brightness maps for the images.
    """
    brightness_maps = []

    for idx, image_path in enumerate(image_paths):
        print(f"Processing image {idx + 1}/{len(image_paths)}")
        # Read the image
        image = cv2.imread(image_path)
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray_image = image.copy()
        h, w = gray_image.shape
        
        # Reshape to a 2D array of pixels
        pixels = gray_image.reshape(-1, 1).astype(np.float32)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)
        clustered = kmeans.labels_.reshape(h, w)
        
        # Save the brightness map
        brightness_maps.append(clustered)

        # Normalize and save the brightness map for visualization
        normalized_map = (clustered / num_clusters * 255).astype(np.uint8)
        image_name = image_path.split('/')[-1].split('.')[0]
        filename = image_save_path + f"/BrightnessMap_{image_name}.png"
        cv2.imwrite(filename, normalized_map)

    return brightness_maps



def generate_brightness_gradient_with_masks(image_paths, brightness_maps, num_bins, masks, image_save_path):
    """
    Computes the Brightness Gradient (Bg) for a list of images using a set of half-disk masks.
    
    Parameters:
        image_paths (list of str): List of image file paths (used for saving filenames).
        brightness_maps (list of np.array): List of Brightness Maps.
        num_bins (int): Number of bins (clusters) in the Brightness Map.
        masks (list of np.array): List of half-disk masks for gradient computation.
        save_prefix (str): Prefix for saved image filenames.
    
    Returns:
        list of np.array: List of Brightness Gradient matrices.
    """
    brightness_gradients = []

    for idx, (image_path, brightness_map) in enumerate(zip(image_paths, brightness_maps)):
        print(f"Processing image {idx + 1}/{len(image_paths)}")
        h, w = brightness_map.shape
        chi_sqr_dist = np.zeros((h, w), dtype=np.float32)

        for i in range(num_bins):
            # Create a binary mask for the current bin
            tmp = (brightness_map == i).astype(np.float32)

            for j in range(0, len(masks), 2):
                # Convolve with pairs of left and right masks
                left_mask = np.array(masks[j])
                right_mask = np.array(masks[j + 1])

                g_i = cv2.filter2D(tmp, -1, left_mask)
                h_i = cv2.filter2D(tmp, -1, right_mask)
                
                # Compute Chi-square gradient
                numerator = (g_i - h_i) ** 2
                denominator = g_i + h_i + 1e-6
                chi_sqr_dist += numerator / denominator
                
        brightness_gradients.append(chi_sqr_dist)

        # Normalize and save the gradient map
        normalized_gradient = (chi_sqr_dist / chi_sqr_dist.max() * 255).astype(np.uint8)

        # Save the color image
        image_name = image_path.split('/')[-1].split('.')[0]
        filename = image_save_path + f"/Bg_{image_name}.png"
        cv2.imwrite(filename, normalized_gradient)

    return brightness_gradients
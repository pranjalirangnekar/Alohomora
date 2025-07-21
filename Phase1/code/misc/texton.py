import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def generate_texton_map(image_paths, filter_bank):
    """
    Applies a filter bank to generate Texton Maps for a set of images.

    Parameters:
        image_paths (list of str): List of image file paths.
        filter_bank (np.array): Filter bank of shape (32, 15, 15).
    
    Returns:
        list of np.array: List of Texton Maps for the input images.
    """
    texton_maps = []

    for image_path in image_paths:
        # Load the image
        image = cv2.imread(image_path)
        
        # Ensure image is not grayscale; convert to grayscale if necessary
        if len(image.shape) == 2:
            grayscale_image = image
        else:
            # grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grayscale_image = image
        
        filtered_responses = []
        
        for filter_ in filter_bank:
            # Apply convolution
            response = cv2.filter2D(grayscale_image, -1, filter_)
            filtered_responses.append(response)
        
        # Stack responses and take max across all filters
        filtered_responses = np.stack(filtered_responses, axis=-1)
        texton_map = np.max(filtered_responses, axis=-1)
        texton_maps.append(texton_map)
    
    return texton_maps

def generate_texture_ids(image_paths, texton_maps, num_clusters, save_image_path="TextonMap"):
    """
    Generates texture IDs using K-means clustering and saves Texton Maps.

    Parameters:
        image_paths (list of str): List of image file paths (used for saving filenames).
        texton_maps (list of np.array): List of Texton Maps.
        num_clusters (int): Number of clusters for K-means.
        save_prefix (str): Prefix for saved image filenames.
    
    Returns:
        list of np.array: List of clustered maps (texture IDs).
    """
    texture_ids = []
    
    for idx, (image_path, texton_map) in enumerate(zip(image_paths, texton_maps)):
        print(f"Processing image {idx + 1}/{len(image_paths)}")
        # Reshape Texton Map for clustering
        h, w, c = texton_map.shape
        reshaped_map = texton_map.reshape(-1, 1)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(reshaped_map)
        
        # Reshape cluster labels back to image size
        clustered_map = kmeans.labels_.reshape(h, w, c)

        # Save Texton Map as image
        image_name = image_path.split('/')[-1].split('.')[0]
        filename = save_image_path + f"/TextonMap_{image_name}.png"
        cv2.imwrite(filename, (clustered_map * (255 // num_clusters)).astype(np.uint8))

        texture_ids.append(clustered_map)
    return texture_ids


def generate_texton_gradient(image_paths, texton_maps, num_bins, masks, image_save_path):
    """
    Computes the Texton Gradient (Tg) for a list of images using a set of half-disk masks.

    Parameters:
        image_paths (list of str): List of image file paths (used for saving filenames).
        texton_maps (list of np.array): List of Texton Maps (from Task 2).
        num_bins (int): Number of bins (clusters) in the Texton Map.
        masks (list of np.array): List of half-disk masks for gradient computation.
        save_prefix (str): Prefix for saved image filenames.
    
    Returns:
        list of np.array: List of Texton Gradient matrices.
    """
    texton_gradients = []

    for idx, (image_path, texton_map) in enumerate(zip(image_paths, texton_maps)):
        print(f"Processing image {idx + 1}/{len(image_paths)}")
        h, w, c = texton_map.shape
        chi_sqr_dist = np.zeros((h, w, c), dtype=np.float32)

        for i in range(num_bins):
            # Create a binary mask for the current bin
            tmp = np.array((texton_map == i).astype(np.float32))

            for j in range(0, len(masks), 2):
                # Convolve with pairs of left and right masks
                left_mask = np.array(masks[j])
                right_mask = np.array(masks[j + 1])
            
                # g_i = convolve(tmp, left_mask, mode='constant', cval=0.0)
                # h_i = convolve(tmp, right_mask, mode='constant', cval=0.0)
                g_i = cv2.filter2D(tmp, -1, left_mask)
                h_i = cv2.filter2D(tmp, -1, right_mask)
                
                # Compute Chi-square gradient
                numerator = (g_i - h_i) ** 2
                denominator = g_i + h_i + 1e-6
                chi_sqr_dist += numerator / denominator

        texton_gradients.append(chi_sqr_dist)

        # Save the gradient map as an image
        gradient_image = (chi_sqr_dist / chi_sqr_dist.max() * 255).astype(np.uint8)
        # Save Texton Map as image
        image_name = image_path.split('/')[-1].split('.')[0]
        filename = image_save_path + f"/Tg_{image_name}.png"
        # cv2.imwrite(filename, (clustered_map * (255 // num_clusters)).astype(np.uint8))
        cv2.imwrite(filename, gradient_image)

    return texton_gradients


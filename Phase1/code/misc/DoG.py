import numpy as np
# from scipy.ndimage import rotate
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt
import math


def gaussian_kernel(size, sigma):
    ##Generate a Gaussian kernel.
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def sobel_filters():
    ##Return Sobel filters for x and y gradients.
    G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return G_x, G_y


# def rotate_img(image, rotation_amount_rad):
# 	# Convert radians to degrees
#     degrees = np.degrees(rotation_amount_rad)
    
#     # Get the image dimensions
#     (h, w) = image.shape[:2]
    
#     # Calculate the center of the image
#     center = (w // 2, h // 2)
    
#     # Compute the rotation matrix
#     rotation_matrix = cv2.getRotationMatrix2D(center, degrees, scale=1.0)
    
#     # Perform the rotation
#     rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
#     return rotated_image

def rotateImage(image, angle, clock_wise = True):
    rows, cols = image.shape[0], image.shape[1]
    # angle = np.rad2deg(angle)
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 2 * (int(clock_wise) - 0.5))
    # fixing the rotation matrix to be n=in the (0 ,0)
    matrix[0, 2] += (matrix[0, 0] + matrix[0, 1] - 1) / 2
    matrix[1, 2] += (matrix[1, 0] + matrix[1, 1] - 1) / 2
    result = cv2.warpAffine(image, matrix, (cols, rows))
    return result

def oriented_dog_filters(scales, orientations, size=10, image_save_path="DoG.png", display=False):
	sobel_x, sobel_y = sobel_filters()
	filter_bank = []
	scales = scales  # Two scales
	orientations = orientations  # Sixteen orientations
	size = size
	for sigma in scales:
		gauss = gaussian_kernel(size, sigma)
		dx = convolve2d(gauss, sobel_x, mode='same')
		dy = convolve2d(gauss, sobel_y, mode='same')
		
		for angle in np.linspace(0, 360, orientations, endpoint=False):
			# Rotate filters to desired angle
			# rotated_dx = rotate(dx, angle, reshape=False)
			# rotated_dy = rotate(dy, angle, reshape=False)
			rotated_dx = rotateImage(dx,angle)
			rotated_dy = rotateImage(dy, angle)

			# Combine filters
			combined_filter = rotated_dx + rotated_dy
			# combined_filter = dx + dy
			filter_bank.append(combined_filter)

	## MATPLOTLIB VISUALIZATION
	rows, cols = len(scales), orientations
	fig, axes = plt.subplots(rows, cols, figsize=(10, 2))

	for i, filt in enumerate(filter_bank):
		ax = axes[i // cols, i % cols]
		ax.imshow(filt, cmap='gray')
		ax.axis('off')
	if display:
		plt.show()
	plt.savefig(image_save_path)

	## OPENCV VISUALIZATION
	# rows, cols = len(scales), orientations
	# filter_height, filter_width = filter_bank[0].shape
	# canvas = np.zeros((rows * filter_height, cols * filter_width), dtype=np.float32)

	# for i, filt in enumerate(filter_bank):
	# 	row = i // cols
	# 	col = i % cols
	# 	y_start, y_end = row * filter_height, (row + 1) * filter_height
	# 	x_start, x_end = col * filter_width, (col + 1) * filter_width
	# 	# Normalize filter to 0-255 range for visualization
	# 	normalized_filter = cv2.normalize(filt, None, 0, 255, cv2.NORM_MINMAX)
	# 	canvas[y_start:y_end, x_start:x_end] = normalized_filter

	# canvas_uint8 = np.uint8(canvas)
	# # Save the image
	# cv2.imwrite(image_save_path, canvas_uint8)

	# # Display the result
	# if display:
	# 	cv2.imshow("Oriented DoG Filters", canvas_uint8)
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()

	return filter_bank
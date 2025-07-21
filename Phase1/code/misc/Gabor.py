import numpy as np
import matplotlib.pyplot as plt
import cv2

def gabor(size, sigma, theta, Lambda, psi, gamma):
    """
    Create a Gabor filter.

    Parameters:
        size (int): The size of the filter (size x size).
        wavelength (float): The wavelength of the sinusoidal component.
        orientation (float): The orientation of the filter in radians.
        sigma (float): The standard deviation of the Gaussian envelope.
        aspect_ratio (float): The aspect ratio of the Gaussian envelope.
        phase (float): The phase offset of the sinusoidal component.

    Returns:
        np.ndarray: The Gabor filter.
    """
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    ## COMMENTED CODE RESULTS IN DYNAMIC SIZE OF FILTER
    # Bounding box
    # nstds = 3  # Number of standard deviation sigma
    # xmax = max(
    #     abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta))
    # )
    # xmax = np.ceil(max(1, xmax))
    # ymax = max(
    #     abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta))
    # )
    # ymax = np.ceil(max(1, ymax))
    # xmin = -xmax
    # ymin = -ymax
    # (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    half_size = size // 2
    y, x = np.meshgrid(
        np.arange(-half_size, half_size + 1),
        np.arange(-half_size, half_size + 1)
    )
    
    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(
        -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
    ) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb

def generate_gabor_filters(num_filters, size, wavelengths, orientations, sigmas, aspect_ratios, phases):
    """
    Generate multiple Gabor filters.

    Parameters:
        num_filters (int): Number of Gabor filters to generate.
        size (int): Size of each filter.
        wavelengths (list): List of wavelengths.
        orientations (list): List of orientations in radians.
        sigmas (list): List of Gaussian sigmas.
        aspect_ratios (list): List of aspect ratios.
        phases (list): List of phase offsets.

    Returns:
        list: A list of Gabor filters.
    """
    filters = []
    for i in range(num_filters):
        filters.append(gabor(
            size = size,
            sigma=sigmas[i % len(sigmas)],
            theta=orientations[i % len(orientations)],
            Lambda=wavelengths[i % len(wavelengths)],
            psi=phases[i % len(phases)],
            gamma=aspect_ratios[i % len(aspect_ratios)],
        ))
    return filters

def visualize_filters(F, save_path, display=False):
    """
    Visualize and save Gabor filters using OpenCV.

    Parameters:
        F (list of np.ndarray): List of Gabor filters.
        save_path (str): Path to save the visualized filters as an image.
        display (bool): Whether to display the filters using OpenCV.
    """
    num_filters = len(F)
    ncols = 8
    nrows = int(np.ceil(num_filters / ncols))
    filter_size = F[0].shape[0]  # Assuming all filters are square and the same size

    # Create a blank canvas to hold all filters
    canvas_height = nrows * filter_size
    canvas_width = ncols * filter_size
    canvas = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255

    # Normalize and place each filter onto the canvas
    for idx, gabor in enumerate(F):
        # Normalize the filter for display
        normalized_gabor = cv2.normalize(gabor, None, 0, 255, cv2.NORM_MINMAX)
        normalized_gabor = np.uint8(normalized_gabor)

        # Compute the position on the canvas
        row = idx // ncols
        col = idx % ncols
        y_start = row * filter_size
        y_end = y_start + filter_size
        x_start = col * filter_size
        x_end = x_start + filter_size

        # Place the filter on the canvas
        canvas[y_start:y_end, x_start:x_end] = normalized_gabor

    # Save the canvas as an image
    cv2.imwrite(save_path, canvas)

    # Optionally display the canvas
    if display:
        cv2.imshow("Gabor Filters", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

## MATPLOTLIB VISUALIZATION
def visualize_filterss(F,save_path, display=False):
# Visualize the Gabor filters
    plt.figure(figsize=(20, 15))
    for i, gabor in enumerate(F):
        plt.subplot(5, 8, i + 1)
        plt.imshow(gabor, cmap='gray')
        plt.axis('off')
        # plt.title(f'Filter {i + 1}')
    plt.tight_layout()
    plt.savefig(save_path)
    if display:
        plt.show()
    else:
        plt.close()

size = 64
num_filters = 40
wavelengths = [10, 15, 20]
orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
sigmas = [5, 8]
aspect_ratios = [0.5, 1.0]
phases = [0, np.pi/2]

# gabor_filters = generate_gabor_filters(num_filters, size, wavelengths, orientations, sigmas, aspect_ratios, phases)
# visualize_filterss(gabor_filters, 'Gabor.png', display=True)

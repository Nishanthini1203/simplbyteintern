import cv2
import numpy as np

def rgb_to_gray(image):
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

def invert_colors(image):
    return 255 - image

def generate_gaussian_kernel(kernel_size, sigma=0.1):
    if sigma == 0:
        sigma = 0.1

    kernel = np.zeros(kernel_size)
    center_x = kernel_size[0] // 2
    center_y = kernel_size[1] // 2

    x, y = np.meshgrid(np.arange(-center_x, center_x + 1), np.arange(-center_y, center_y + 1))
    exponent = -((x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = (1 / (2 * np.pi * sigma ** 2)) * np.exp(exponent)

    kernel /= np.sum(kernel)
    return kernel

def apply_convolution(image, kernel):
    kernel_size = kernel.shape
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel_size

    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    padded_image = np.pad(image, ((padding_height, padding_height), (padding_width, padding_width)), mode='edge')
    output_image = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            image_patch = padded_image[i:i+kernel_height, j:j+kernel_width]
            convolved_value = np.sum(image_patch * kernel)
            output_image[i, j] = convolved_value

    return output_image

def edge_detection(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gradient_x = apply_convolution(image, sobel_x)
    gradient_y = apply_convolution(image, sobel_y)

    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    return gradient_magnitude, gradient_direction

def apply_threshold(image, threshold):
    binary_image = np.zeros_like(image)
    binary_image[image >= threshold] = 255
    return binary_image

def dilate(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, dtype=np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image

# Load the input image
input_image = cv2.imread('C:/Users/Nisha/Desktop/IMG-20180805-WA0001.jpg')

# Convert the input image to grayscale
gray_image = rgb_to_gray(input_image)

# Invert the grayscale image
inverted_image = invert_colors(gray_image)

# Apply Gaussian blur
kernel_size = (5, 5)
sigma = 1.0
gaussian_kernel = generate_gaussian_kernel(kernel_size, sigma)
blurred_image = apply_convolution(inverted_image, gaussian_kernel)

# Perform edge detection
edges, _ = edge_detection(blurred_image)

# Apply thresholding to obtain a binary image
threshold = 150
thresholded_image = apply_threshold(edges, threshold)

# Dilate the binary image
dilated_image = dilate(thresholded_image)

# Save the final pencil sketch image
output_path = 'C:/Users/Nisha/Desktop/pencil_sketch.jpg'  # Specify the output file path
cv2.imwrite(output_path, dilated_image)
print("Pencil sketch saved successfully!")

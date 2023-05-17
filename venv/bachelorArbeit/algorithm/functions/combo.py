import cv2
import numpy as np
import os
from scipy.ndimage import median_filter
from skimage import io, exposure
from PIL import Image, ImageOps, ImageTk, ImageFilter
from medianFilterFunc import apply_median_filter, apply_sharpen_filter
from kMeansClusteringFunc import kMeans_segment_image
from calculatePerimeter import calculate_perimeter
from calculateArea import calculate_area

# Load the image
originalImage = cv2.imread("../../images/ctisus/ctisusBmp/adrenal_1-01.bmp")

# Apply the median filter
medianFilteredImage = apply_median_filter(originalImage, 7)

# Apply the sharpen filter
sharpenFilteredImage = apply_sharpen_filter(medianFilteredImage)
sharpenFilteredImageArr = np.array(sharpenFilteredImage)

# Apply k-Means Clustering
segments = kMeans_segment_image(sharpenFilteredImageArr, 6)

# Directory to save the segmented images
output_dir = './segmentedImages/'

# Display the results
cv2.imshow('Original Image', originalImage)
cv2.imshow('Filtered Image', sharpenFilteredImageArr)

for i, segment in enumerate(segments):
    segment_filename = f'segment_{i}.bmp'
    segment_path = os.path.join(output_dir, segment_filename)
    cv2.imwrite(segment_path, segment)
    cv2.imshow(f'Segment {i}', segment)
cv2.waitKey(0)
cv2.destroyAllWindows()

regionOfInterest = './segmentedImages/segment_1.bmp'

perimeter = calculate_perimeter(regionOfInterest)
print('Perimeter:', perimeter)

area = calculate_area(regionOfInterest)
print('Area:', area)

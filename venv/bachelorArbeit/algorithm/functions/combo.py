import cv2
import numpy as np
from scipy.ndimage import median_filter
from skimage import io, exposure
from PIL import Image, ImageOps, ImageTk, ImageFilter
from medianFilterFunc import apply_median_filter, apply_sharpen_filter
from kMeansClusteringFunc import kMeans_segment_image

# Load the image
originalImage = cv2.imread("../../images/ctisus/ctisusBmp/adrenal_1-01.bmp")

# Apply the median filter
medianFilteredImage = apply_median_filter(originalImage, 7)

# Apply the sharpen filter
sharpenFilteredImage = apply_sharpen_filter(medianFilteredImage)
sharpenFilteredImageArr = np.array(sharpenFilteredImage)

# Apply k-Means Clustering
segments = kMeans_segment_image(sharpenFilteredImageArr, 6)

# Display the results
cv2.imshow('Original Image', originalImage)
cv2.imshow('Filtered Image', sharpenFilteredImageArr)

for i, segment in enumerate(segments):
    cv2.imshow(f'Segment {i}', segment)
cv2.waitKey(0)
cv2.destroyAllWindows()

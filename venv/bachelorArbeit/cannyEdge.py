import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load the image
image = cv2.imread('./images/ctisus/ctisusBmp/adrenal_1-01.bmp')

# Convert the image to a flattened array
X = image.reshape(-1, 3)

# Perform k-means clustering
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)

# Assign each pixel to the cluster with the closest centroid
labels = kmeans.predict(X)

# Create a segmentation map
segmentation_map = labels.reshape(image.shape[:2])

# Apply a binary threshold to the segmentation map
_, thresholded_map = cv2.threshold(segmentation_map.astype(np.uint8), 2, 255, cv2.THRESH_BINARY)

# Perform Canny edge detection on the thresholded map
edges = cv2.Canny(thresholded_map, 100, 200)

# Display the original image, thresholded map, and edges
cv2.imshow('Original Image', image)
cv2.imshow('Thresholded Map', thresholded_map)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
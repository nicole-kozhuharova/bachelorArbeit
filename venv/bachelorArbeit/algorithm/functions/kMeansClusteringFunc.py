import cv2
import numpy as np

def kMeans_segment_image(image, num_clusters):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reshape the image to a 2D array of pixels and convert it to float32 type
    pixel_values = gray.reshape((-1, 1)).astype(np.float32)

    # Perform k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(pixel_values, num_clusters, None, criteria, 10, flags)

    # Reshape the labels to the shape of the original image
    segmented = labels.reshape(gray.shape)

    # Create masks for each cluster
    masks = []
    for i in range(num_clusters):
        mask = np.zeros_like(segmented)
        mask[segmented == i] = 255
        masks.append(mask)

    # Apply each mask to the original image to extract segments
    segments = []
    for mask in masks:
        segment = cv2.bitwise_and(image, image, mask=mask.astype('uint8'))
        segments.append(segment)

    # Return the segments
    return segments

# Example usage
# image_path = '../../images/ctisus/ctisusBmp/adrenal_1-01.bmp'
# # Load the image and convert it to grayscale
# image = cv2.imread(image_path)
# segments = kMeans_segment_image(image, 7)
#
# # Display the results
# original_image = cv2.imread(image_path)
# cv2.imshow('Original Image', original_image)
# for i, segment in enumerate(segments):
#     cv2.imshow(f'Segment {i}', segment)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

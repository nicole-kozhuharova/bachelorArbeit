import cv2
import numpy as np
from scipy.ndimage import median_filter
from PIL import Image
from plotImages import plotImage

# Load the image
img = np.array(Image.open("../images/ctisus/ctisusTiff/adrenal_1-07.tiff"))

# Apply the median filter
size = 3
img_filtered = median_filter(img, size=size)

# Display the original and median-filtered images
plotImage(img, img_filtered)

import cv2
from PIL import Image
import numpy as np
from plotImages import plotImage
from scipy.ndimage import median_filter

######## Lookup table with median filter

# # Load the image
# img = Image.open('./images/ctisus/ctisusTiff/adrenal_1-01.tiff')
#
# # Create a LUT to increase the image brightness by 50%
# lut = np.arange(256) + 0.2 * 255
#
# # Apply the LUT to the image
# result = img.point(lut)
#
# # Apply the median filter
# size = 3
# img_filtered = median_filter(result, size=size)
#
# # Display the original and median-filtered images
# plotImage(img, img_filtered)




#3 Load the image
img = Image.open('./images/ctisus/ctisusTiff/adrenal_1-01.tiff')

# Apply the median filter
size = 3
img_filtered = median_filter(img, size=size)

# Create a LUT to increase the image brightness by 50%
lut = np.arange(256) + 0.2 * 255

# Apply the LUT to the image
pil_img_filtered = Image.fromarray(img_filtered)
result = pil_img_filtered.point(lut)

plotImage(img, result)

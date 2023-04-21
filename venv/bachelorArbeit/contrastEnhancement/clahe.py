import cv2
from plotImages import plotImage

# Load the image in grayscale
img = cv2.imread('../images/ctisus/ctisusTiff/adrenal_1-01.tiff', 0)

# Create a CLAHE object with specified parameters
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Apply CLAHE to the image
clahe_img = clahe.apply(img)

# Display the original and processed images
plotImage(img, clahe_img)


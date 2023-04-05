import pydicom
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load DICOM image
dcm_image = pydicom.dcmread('./images/dicomImages/1-02.dcm')

# Convert DICOM to Pillow image
pil_image = Image.fromarray(dcm_image.pixel_array)

# Convert Pillow image to NumPy array
np_image = np.array(pil_image)

# Save image as TIFF
plt.imsave('./images/jpegImages/1-02.tiff', np_image, cmap='gray')

# Display the TIFF image using matplotlib
plt.imshow(np_image, cmap='gray')
plt.show()

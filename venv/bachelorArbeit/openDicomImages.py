import pydicom
import matplotlib.pyplot as plt
from PIL import Image

# Load the DICOM file
ds = pydicom.dcmread('./images/dicomImages/101.dcm')

# Extract the pixel data
pixel_data = ds.pixel_array

# Display the image
plt.imshow(pixel_data, cmap=plt.cm.gray)
plt.show()

# Convert to image to 8-bit grayscale and save as JPEG
# Because JPEG format supports only 8-bit and 24-bit color images, not 16-bit grayscale
image = Image.fromarray(pixel_data.astype('uint8'), mode='L')
image.save('./images/jpegImages/101.jpeg')
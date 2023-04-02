import pydicom
import matplotlib.pyplot as plt
from PIL import Image

def convert_to_jpeg(dicom_path, jpeg_path):
    # Load DICOM file
    ds = pydicom.dcmread(dicom_path)

    # Extract pixel array
    pixel_data = ds.pixel_array

    # Display the image
    plt.imshow(pixel_data, cmap=plt.cm.gray)
    plt.show()

    # Convert to 8-bit grayscale
    image = Image.fromarray(pixel_data.astype('uint8'), mode='L')

    # Save as JPEG
    image.save(jpeg_path)



convert_to_jpeg('./images/dicomImages/1-02.dcm', './images/jpegImages/1-02.jpeg')

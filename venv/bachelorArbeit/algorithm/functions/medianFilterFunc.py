import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk, ImageFilter

def apply_median_filter(img, ksize):
    # Apply the median filter
    img = cv2.medianBlur(img, ksize)

    # Convert the NumPy array to a PIL image
    # pil_img = Image.fromarray(img)

    # return pil_img
    return img

def apply_sharpen_filter(image):
    # Convert the NumPy array to a PIL image so that we can apply filter(ImageFilter.SHARPEN)
    pil_img = Image.fromarray(image)

    # Apply the sharpen filter
    sharpen = pil_img.filter(ImageFilter.SHARPEN)

    return sharpen

# # Image path
# image_path = "../../images/ctisus/ctisusBmp/adrenal_1-01.bmp"
# # Load the image
# img = cv2.imread(image_path)
#
# # Apply the median filter
# median_filtered_img = apply_median_filter(img, 7)
# # pil_img = Image.fromarray(median_filtered_img)
# # Apply the sharpen filter
# # sharpen_filtered_img = apply_sharpen_filter(pil_img)
# sharpen_filtered_img = apply_sharpen_filter(median_filtered_img)
#
#
# # Convert the PIL images to NumPy arrays for displaying with OpenCV
# median_filtered_array = np.array(median_filtered_img)
# sharpen_filtered_array = np.array(sharpen_filtered_img)
#
# # Display the original and filtered images
# cv2.imshow('Original Image', cv2.imread(image_path))
# cv2.imshow('Median Filtered Image', median_filtered_array)
# cv2.imshow('Sharpen Filtered Image', sharpen_filtered_array)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

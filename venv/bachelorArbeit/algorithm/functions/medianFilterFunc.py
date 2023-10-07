import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk, ImageFilter

def apply_median_filter(img, ksize):
    # Apply the median filter
    img = cv2.medianBlur(img, ksize)

    return img

def apply_sharpen_filter(image):
    # Convert the NumPy array to a PIL image so that we can apply filter(ImageFilter.SHARPEN)
    pil_img = Image.fromarray(image)

    # Apply the sharpen filter
    sharpen = pil_img.filter(ImageFilter.SHARPEN)

    return sharpen


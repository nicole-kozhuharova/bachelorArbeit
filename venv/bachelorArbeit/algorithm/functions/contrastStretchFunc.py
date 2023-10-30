from skimage import io, exposure
import cv2
import numpy as np

def apply_contrast_stretching(img):

    # Apply contrast stretching
    p2, p98 = np.percentile(img, (80, 99))
    stretched_image = exposure.rescale_intensity(img, in_range=(p2, p98))
    rgb_stretched_image = cv2.cvtColor(stretched_image, cv2.COLOR_BGR2RGB)

    cv2.imshow('stretched image', stretched_image)
    # cv2.imshow('rgb stretched image', rgb_stretched_image)

    # Wait for key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return rgb_stretched_image
    return stretched_image


# cv2.imshow('stretched image', stretched_image)
# # Wait for key press
# cv2.waitKey(0)
# cv2.destroyAllWindows()
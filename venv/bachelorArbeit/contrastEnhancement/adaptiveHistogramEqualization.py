from skimage import io, exposure
import matplotlib.pyplot as plt

# Load a medical image in tif format
img = io.imread('../images/jpegImages/1-15.tiff')

# Perform adaptive histogram equalization
img_eq = exposure.equalize_adapthist(img, clip_limit=0.03)

# Save the enhanced image
# io.imsave('../images/jpegImages/1-15-adaptHist.tiff', img_eq)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_eq, cmap='gray')
plt.title('Equalized Image'), plt.xticks([]), plt.yticks([])
plt.show()
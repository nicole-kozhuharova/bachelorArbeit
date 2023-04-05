from skimage import io, exposure
import matplotlib.pyplot as plt

# Load a medical image in tif format
img = io.imread('../images/jpegImages/1-15.tiff')

# Perform gamma correction
gamma_corrected = exposure.adjust_gamma(img, gamma=0.5)

# Save the gamma corrected image
io.imsave('../images/jpegImages/1-15-gammaCorrection.tiff', gamma_corrected)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(gamma_corrected, cmap='gray')
plt.title('Equalized Image'), plt.xticks([]), plt.yticks([])
plt.show()
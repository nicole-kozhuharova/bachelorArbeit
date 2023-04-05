from skimage import io, exposure
import matplotlib.pyplot as plt
from plotImages import plotImage

# Load a medical image in tif format
img = io.imread('../images/jpegImages/1-15.tiff')

# Perform gamma correction
gamma_corrected = exposure.adjust_gamma(img, gamma=0.5)

# Save the gamma corrected image
io.imsave('../images/jpegImages/1-15-gammaCorrection.tiff', gamma_corrected)

plotImage(img, gamma_corrected)
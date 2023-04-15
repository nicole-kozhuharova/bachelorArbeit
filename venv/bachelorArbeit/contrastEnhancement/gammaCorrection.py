from skimage import io, exposure
import matplotlib.pyplot as plt
from plotImages import plotImage

# Load a medical image in tif format
img = io.imread('../images/ctisus/ctisusTiff/adrenal_1-07.tiff')

# Perform gamma correction
gamma_corrected = exposure.adjust_gamma(img, gamma=0.5)

# Save the gamma corrected image
# io.imsave('../images/PETjpegImages/1-01.jpg', gamma_corrected)

plotImage(img, gamma_corrected)
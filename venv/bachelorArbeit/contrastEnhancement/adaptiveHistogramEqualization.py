from skimage import io, exposure
import matplotlib.pyplot as plt
from plotImages import plotImage

# Load a medical image in tif format
img = io.imread('../images/jpegImages/pet-1-01.tiff')

# Perform adaptive histogram equalization
img_eq = exposure.equalize_adapthist(img, clip_limit=0.03)

# Save the enhanced image
# io.imsave('../images/jpegImages/1-15-adaptHist.tiff', img_eq)

plotImage(img, img_eq)
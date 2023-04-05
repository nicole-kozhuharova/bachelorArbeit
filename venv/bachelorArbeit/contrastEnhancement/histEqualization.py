import cv2
import matplotlib.pyplot as plt

def plot_histogram_equalized_image(image_path):
    # read the image
    img = cv2.imread(image_path, 0)

    # perform histogram equalization
    equ = cv2.equalizeHist(img)

    # display the original and equalized images
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(equ, cmap='gray')
    plt.title('Equalized Image'), plt.xticks([]), plt.yticks([])
    plt.show()

plot_histogram_equalized_image('./images/jpegImages/1-15.tiff')

#In this example, we use the OpenCV library to read the input image
# and perform histogram equalization using the cv2.equalizeHist() function.
# We then use Matplotlib to display the original and equalized images side-by-side.
# The cmap='gray' argument specifies that the images should be displayed in grayscale.

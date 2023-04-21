from PIL import Image

# Open the JPEG image
jpeg_image = Image.open('./images/ctisus/ctisusJpeg/adrenal_1-01.jpg')

# Convert the image to BMP format
bmp_image = jpeg_image.convert('RGB')

# Save the BMP image
bmp_image.save('./images/ctisus/ctisusBmp/adrenal_1-01.bmp')

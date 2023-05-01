## doesnt work but could give it a try
# from skimage import io, segmentation, color
# import numpy as np
#
# # Load the image
# img = io.imread('./images/ctisus/ctisusBmp/adrenal_1-01.bmp')
#
# # Define the seed point
# seed = (220, 260)
#
# # Perform region growing
# labels = segmentation.flood_fill(img, seed_point=tuple(np.flip(seed)), new_value=255, connectivity=1)
#
# # Colorize the regions
# label_image = color.label2rgb(labels, image=img)
#
# # Display the results
# io.imshow(label_image)
# io.show()








# import skimage.segmentation as seg
# import skimage.filters as filters
# from skimage import io, color
#
# # Load the image
# img = io.imread("./images/ctisus/ctisusBmp/adrenal_1-01.bmp")
#
# img = color.rgb2gray(img)
#
# # Apply a threshold to the image to create a mask
# mask = img > filters.threshold_otsu(img)
#
# # Use the mask to split the image into regions
# regions = seg.felzenszwalb(img, scale=200, sigma=0.5, min_size=50)
#
# # Display the segmented image
# io.imshow(regions)
# io.show()








############################################################ square i might need for coordinates
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import io, color, filters, morphology, restoration, segmentation
#
# # Load the medical image
# image = io.imread('./images/ctisus/ctisusBmp/adrenal_1-01.bmp')
#
# # Pre-process the image
# image = color.rgb2gray(image)
# image = restoration.denoise_tv_chambolle(image, weight=0.1)
# image = filters.median(image, selem=morphology.disk(3))
# image = filters.rank.mean(image, selem=morphology.disk(5))
# image = filters.sobel(image)
#
# # Define the ROI
# roi = np.zeros_like(image)
# roi[200:350, 150:350] = 1
#
# # Initialize the Split Operator algorithm
# num_iterations = 100
# reg_param = 0.1
# model = np.zeros_like(image)
# model[roi == 1] = 1
# gamma = np.zeros_like(image)
# f = np.zeros_like(image)
#
# # Apply the Split Operator algorithm
# for i in range(num_iterations):
#     # Update the model
#     model[roi == 1] = (image + gamma)[roi == 1] / (1 + reg_param)
#     # Update the segmentation map
#     f = (image + gamma - model) > 0
#     # Update the Lagrange multiplier
#     gamma += image - model - f
#
# # Extract the segmentation map
# segmentation_map = f.astype(int)
#
# # Post-process the segmentation map
# segmentation_map = morphology.remove_small_objects(segmentation_map, min_size=100)
#
# # Visualize the segmentation map
# fig, ax = plt.subplots()
# ax.imshow(image, cmap='gray')
# ax.imshow(segmentation_map, cmap='jet', alpha=0.3)
# plt.show()












# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
# def regrow(full_file, t, s):
#     I = cv2.imread(full_file, cv2.IMREAD_COLOR)
#     I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
#     I = I.astype(np.float32) / 255.0
#     cv2.imshow("Input Image", I)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     print("Select a seed point of the region you want to segment.")
#     fig, ax = plt.subplots()
#     ax.imshow(I)
#     point = plt.ginput(1)[0]
#     plt.close(fig)
#     p2, p1 = int(point[1]), int(point[0])
#     print(f"Seed point: ({p1}, {p2}, {I.shape}, {point})")
#
#     # free memory for a region
#     R = np.zeros_like(I[:, :, 0])  # output
#     im_size = I.shape[:2]
#     ival_reg = I[p1, p2, 0]  # average intensity value of the region
#     reg_size = 1
#     # free memory for neighbor pixels
#     pos = 0
#     list = np.zeros((50000, 3), dtype=int)
#     pix_diff = 0
#     # neighbour candidates for the region
#     cand = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
#
#     # do the region growing until the intensity difference between the region mean and a
#     # new pixel become larger than a certain threshold
#     while (pix_diff < t) and (reg_size < I.size):
#         for j in range(4):
#             n1 = p1 + cand[j, 0]
#             n2 = p2 + cand[j, 1]
#             # check if the neighbor is in or out of the image
#             ins = (n1 >= 0) and (n2 >= 0) and (n1 < im_size[0]) and (n2 < im_size[1])
#             if (ins) and (R[n1, n2] == 0):
#                 pos = pos + 1
#                 list[pos, :] = [n1, n2, I[n1, n2, 0] * 255.0]
#                 R[n1, n2] = 1
#
#         # new block of memory
#         if (pos + 10 > 10000):
#             list[(pos + 1):20000, :] = 0
#
#         diff = np.abs(list[1:pos, 2] - ival_reg)
#         index = np.argmin(diff)
#         pix_diff = diff[index]
#         R[p1, p2] = 1
#         reg_size = reg_size + 1
#         ival_reg = (ival_reg * reg_size + list[index, 2]) / (reg_size + 1)
#         p1 = list[index, 0]
#         p2 = list[index, 1]
#         list[index, :] = list[pos, :]
#         pos = pos - 1
#
#     # display the segmented region
#     cv2.imshow("Segmented Region", R)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     # Convert binary mask to 3-dimensional array with the same shape as I
#     R_3d = np.repeat(np.expand_dims(R, axis=-1), 3, axis=-1)
#
#     # display original image+region
#     cv2.imshow("Original Image + Region", I + R_3d)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     if s != 0:
#         pathstr, name = os.path.split(full_file)
#         name, ext = os.path.splitext(name)
#         newImagePath = os.path.join(pathstr, f"{name}_RGsegm_{t}{ext}")
#         cv2.imwrite(newImagePath, R * 255)
#
#         newImagePath = os.path.join(pathstr, f"{name}_RGsegm+orig_{t}{ext}")
#         cv2.imwrite(newImagePath, (I + R) * 255)
#
#     # convert I
#
#
# regrow('./images/ctisus/ctisusBmp/adrenal_1-01.bmp', 20, 0)








# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# def regiongrowing(I, x=None, y=None, reg_maxdist=0.2):
#     if y is None:
#         plt.imshow(I, cmap='gray')
#         plt.axis('off')
#         x, y = plt.ginput(2, timeout=10000)
#         x, y = int(round(x[0])), int(round(y[0]))
#     J = np.zeros_like(I)
#     Isizes = I.shape
#     reg_mean = I[x,y]
#     reg_size = 1
#     neg_free = 10000
#     neg_pos = 0
#     neg_list = np.zeros((neg_free, 3))
#     pixdist = 0
#     neigb = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
#     while pixdist < reg_maxdist and reg_size < np.prod(Isizes):
#         for j in range(4):
#             xn = x + neigb[j, 0]
#             yn = y + neigb[j, 1]
#             ins = (xn >= 0 and yn >= 0 and xn < Isizes[0] and yn < Isizes[1])
#             if ins and J[xn, yn] == 0:
#                 neg_pos += 1
#                 neg_list[neg_pos, :] = [xn, yn, I[xn, yn]]
#                 J[xn, yn] = 1
#         if neg_pos + 10 > neg_free:
#             neg_free += 10000
#             neg_list.resize((neg_free, 3))
#             neg_list[neg_pos + 1:, :] = 0
#         dist = np.abs(neg_list[:neg_pos, 2] - reg_mean)
#         pixdist, index = np.min(dist), np.argmin(dist)
#         J[x, y] = 2
#         reg_size += 1
#         reg_mean = (reg_mean * reg_size + neg_list[index, 2]) / (reg_size + 1)
#         x, y = int(neg_list[index, 0]), int(neg_list[index, 1])
#         neg_list[index, :] = neg_list[neg_pos, :]
#         neg_pos -= 1
#     return J.astype(np.bool)
#
# I = cv2.imread('./images/ctisus/ctisusBmp/adrenal_1-01.bmp', cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
# plt.imshow(I, cmap='gray')
# plt.axis('off')
# plt.show()
# J = regiongrowing(I)
# plt.imshow(I + J, cmap='gray')
# plt.axis('off')
# plt.show()




# ## GUI from Matlab
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from collections import deque
#
# def regrow(full_file, t, s):
#     I = cv2.imread(full_file, cv2.IMREAD_COLOR)
#     I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
#     I = I.astype(np.float32) / 255.0
#     cv2.imshow("Input Image", I)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     print("Select a seed point of the region you want to segment.")
#     fig, ax = plt.subplots()
#     ax.imshow(I)
#     point = plt.ginput(1)[0]
#     plt.close(fig)
#     p2, p1 = int(point[1]), int(point[0])
#     print(f"Seed point: ({p1}, {p2}, {I.shape}, {point})")
#
#     # free memory for a region
#     R = np.zeros_like(I[:, :, 0])  # output
#     im_size = I.shape[:2]
#     ival_reg = I[p1, p2, 0]  # average intensity value of the region
#     reg_size = 1
#     # free memory for neighbor pixels
#     pos = 0
#     q = deque(maxlen=20000)
#     pix_diff = 0
#     # neighbour candidates for the region
#     cand = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
#
#     # do the region growing until the intensity difference between the region mean and a
#     # new pixel become larger than a certain threshold
#     while (pix_diff < t) and (reg_size < I.size):
#         for j in range(4):
#             n1 = p1 + cand[j, 0]
#             n2 = p2 + cand[j, 1]
#             # check if the neighbor is in or out of the image
#             ins = (n1 >= 0) and (n2 >= 0) and (n1 < im_size[0]) and (n2 < im_size[1])
#             if (ins) and (R[n1, n2] == 0):
#                 pos = pos + 1
#                 q.append([n1, n2, I[n1, n2, 0] * 255.0])
#                 R[n1, n2] = 1
#
#         diff = np.abs([x[2] for x in q] - ival_reg)
#         index = np.argmin(diff)
#         pix_diff = diff[index]
#         R[p1, p2] = 1
#         reg_size = reg_size + 1
#         ival_reg = (ival_reg * reg_size + q[index][2]) / (reg_size + 1)
#         p1 = q[index][0]
#         p2 = q[index][1]
#         q.remove(q[index])
#         pos = len(q)
#
#     # display the segmented region
#     cv2.imshow("Segmented Region", R)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     # Convert binary mask to 3-dimensional array with the same shape as I
#     R_3d = np.repeat(np.expand_dims(R, axis=-1), 3, axis=-1)
#
#     # display original image+region
#     cv2.imshow("Original Image + Region", I + R_3d)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     if s != 0:
#         pathstr, name = os.path.split(full_file)
#         name, ext = os.path.splitext(name)
#         newImagePath = os.path.join(pathstr, f"{name}_RGsegm_{t}{ext}")
#         cv2.imwrite(newImagePath, R * 255)
#
#         newImagePath = os.path.join(pathstr, f"{name}_RGsegm+orig_{t}{ext}")
#         cv2.imwrite(newImagePath, (I + R) * 255)
#
# regrow('./images/ctisus/ctisusBmp/adrenal_1-01.bmp', 20, 0)













# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define the image and the seed point
# img = plt.imread('./images/ctisus/ctisusBmp/adrenal_1-01.bmp')
# seed = (100, 100)
#
# # Define the connectivity criterion (4 or 8 neighbors)
# connectivity = 4
#
# # Define the initial threshold value
# threshold = img[seed]
#
# # Initialize the region
# region = np.zeros_like(img, dtype=bool)
# region[seed] = True
#
# # Define a function to check if a pixel is already part of the region or not
# def check_pixel(pixel):
#     neighbors = get_neighbors(pixel, connectivity)
#     neighbor_rows, neighbor_cols = zip(*neighbors)
#     return np.any((abs(img[pixel[0], pixel[1], :3] - threshold) < 0.1) & region[neighbor_rows, neighbor_cols])
#
# # Define a function to get the neighbors of a pixel
# def get_neighbors(pixel, connectivity):
#     # if connectivity == 4:
#     #     neighbors = [(pixel[0] + 1, pixel[1]), (pixel[0] - 1, pixel[1]), (pixel[0], pixel[1] + 1), (pixel[0], pixel[1] - 1)]
#     # elif connectivity == 8:
#     #     neighbors = [(pixel[0] + 1, pixel[1]), (pixel[0] - 1, pixel[1]), (pixel[0], pixel[1] + 1), (pixel[0], pixel[1] - 1),
#     #             (pixel[0] + 1, pixel[1] + 1), (pixel[0] - 1, pixel[1] - 1), (pixel[0] + 1, pixel[1] - 1), (pixel[0] - 1, pixel[1] + 1)]
#     if connectivity == 4:
#         neighbors = [(pixel[0] + 1, pixel[1]), (pixel[0], pixel[1]), (pixel[0], pixel[1] ), (pixel[0], pixel[1] - 1)]
#     elif connectivity == 8:
#         neighbors = [(pixel[0] + 1, pixel[1]), (pixel[0] - 1, pixel[1]), (pixel[0], pixel[1] + 1), (pixel[0], pixel[1] - 1),
#                 (pixel[0] + 1, pixel[1] + 1), (pixel[0] - 1, pixel[1] - 1), (pixel[0] + 1, pixel[1] - 1), (pixel[0] - 1, pixel[1] + 1)]
#     print(neighbors)
#     return neighbors
#
# # Iterate through the image and add connected pixels to the region
# while True:
#     added = False
#     for x in range(img.shape[0]):
#         for y in range(img.shape[1]):
#             pixel = (x, y)
#             if check_pixel(pixel) and not region[pixel]:
#                 region[pixel] = True
#                 added = True
#     if not added:
#         break
#
# plt.imshow(region, cmap='gray')
# plt.show()


import cv2
import numpy as np

def get8n(x, y, shape):
    out = []
    maxx = shape[1]-1
    maxy = shape[0]-1

    #top left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top center
    outx = x
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #left
    outx = min(max(x-1,0),maxx)
    outy = y
    out.append((outx,outy))

    #right
    outx = min(max(x+1,0),maxx)
    outy = y
    out.append((outx,outy))

    #bottom left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom center
    outx = x
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    return out

def region_growing(img, seed):
    list = []
    outimg = np.zeros_like(img)
    list.append((seed[0], seed[1]))
    processed = []
    while(len(list) > 0):
        pix = list[0]
        outimg[pix[0], pix[1]] = 255
        for coord in get8n(pix[0], pix[1], img.shape):
            if img[coord[0], coord[1]] != 0:
                outimg[coord[0], coord[1]] = 255
                if not coord in processed:
                    list.append(coord)
                processed.append(coord)
        list.pop(0)
        cv2.imshow("progress",outimg)
        cv2.waitKey(1)
    return outimg

def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print ('Seed: ' + str(x) + ', ' + str(y), img[y,x])
        clicks.append((y,x))

clicks = []
image = cv2.imread('./images/ctisus/ctisusBmp/adrenal_1-01.bmp', 0)
ret, img = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
cv2.namedWindow('Input')
cv2.setMouseCallback('Input', on_mouse, 0, )
cv2.imshow('Input', image)
cv2.waitKey()
seed = clicks[-1]
out = region_growing(img, seed)
cv2.imshow('Region Growing', out)
cv2.waitKey()
cv2.destroyAllWindows()

# source: https://github.com/zjgirl/RegionGrowing-1/blob/master/RegionGrowing.py








































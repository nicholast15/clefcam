import skimage as ski
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage import io, exposure, restoration, transform
from skimage import data, color

import numpy as np

from skimage.transform import hough_line, hough_line_peaks, hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage.draw import circle_perimeter, ellipse_perimeter

from matplotlib import cm
from skimage.util import img_as_ubyte

from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

from skimage import morphology, measure

import array

def staff_lines(image):

    thresh = threshold_otsu(image)
    binary = image > thresh

    binary = ski.util.invert(binary)

    #horizontal lines
    tested_angles_hor = np.linspace(-np.pi, 0, 180, endpoint=False)
    h_hor, theta_hor, d_hor = hough_line(binary, theta=tested_angles_hor)
    ycoord = []
    
    for _hor, angle_hor, dist_hor in zip(*hough_line_peaks(h_hor, theta_hor, d_hor,min_distance=3)):
        ycoord.append(dist_hor)

    ycoord_abs = [int(abs(element)) for element in ycoord]

    return ycoord_abs

def bar_tail_lines(image):
    tested_angles = np.linspace(-0.01, 0.01, 2, endpoint=False)
    h, theta, d = ski.transform.hough_line(image, theta=tested_angles)

    accum, angles, dist = ski.transform.hough_line_peaks(h, theta, d, min_distance=10)
    dist = dist.astype(np.int32)
    return dist

def blob():
    #image = ski.color.rgb2gray(ski.color.rgba2rgb(io.imread("Img\TEST_MUSIC_single_line_nowriting_V2.png")))
    #image = ski.color.rgb2gray(ski.color.rgba2rgb(io.imread("Img/fur_elise_one_line.png")))
    image = ski.color.rgb2gray(ski.color.rgba2rgb(io.imread("Img/halfnote.png")))
    #image = ski.color.rgb2gray(ski.color.rgba2rgb(io.imread("Img/fullNote_temp_oneline.png")))


    denoised = restoration.denoise_nl_means(image, h=0.1, fast_mode=True)
    enhanced = exposure.equalize_adapthist(denoised)

    thresh = threshold_otsu(enhanced)
    image_gray = image > thresh

    blobs_doh = blob_doh(image_gray, min_sigma = 5, max_sigma=20, threshold=0.02)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(image)
    ax[1].set_title('Blobs')
    ax[1].imshow(image)

    for i in range(len(blobs_doh[:,0])):
        c = plt.Circle((blobs_doh[i,1], blobs_doh[i,0]), blobs_doh[i,2], color='red', linewidth=2, fill=False)
        ax[1].add_patch(c)

    plt.tight_layout()
    plt.show()
    
    return blobs_doh

#blob()

def houghcircles(image, ycoord, xcoord):

    height, width = image.shape
    lines = np.ones((height, width), dtype=np.uint8) * 255
    for i in range(len(ycoord)):
        for j in range(width):
            lines[ycoord[i]][j] = 0

    for i in range(len(xcoord)):
        for j in range(height):
            lines[j][xcoord[i]] = 0

    image_nolines = ski.util.invert(np.clip(ski.util.invert(image) - ski.util.invert(lines),0,255))

    denoised = restoration.denoise_nl_means(image_nolines, h=0.05, fast_mode=True)
    enhanced = exposure.equalize_adapthist(denoised)

    thresh = threshold_otsu(enhanced)
    binary = enhanced > thresh

    #morph = morphology.area_opening(binary,4,8)
    morph = morphology.area_closing(binary,8,8)

    thresh = threshold_otsu(morph)
    morph = morph > thresh
    
    edges = canny(morph, sigma=2.0, low_threshold=0.1, high_threshold=0.5)

    hough_radii = np.arange(4, 10, 1, dtype=int)
    hough_res = hough_circle(edges, hough_radii)

    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,min_xdistance=15, min_ydistance=10, threshold = 0.3)


    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image = color.gray2rgb(image)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
        image[circy, circx] = (255, 0, 0)

    ax.imshow(image, cmap=plt.cm.gray)
    plt.show()
    return cx, cy, radii

def note_type(image, x, y, radius, vert_xcoord):
    tot = 0
    count = 0
    for i in range(np.floor(radius/2)*2+1):
        for j in range(np.floor(radius/2)*2+1):
            tot += image[y + i - np.floor(radius/2),x + j - np.floor(radius/2)]
            count += 1

    if tot/count > 0.5:
        return 4
    else:
        for i in range(len(vert_xcoord)):
            if vert_xcoord[i] > np.floor(x - radius*2) or vert_xcoord[i] < np.floor(x + radius*2):
                return 2
            else:
                return 1

#image = ski.color.rgb2gray(ski.color.rgba2rgb(io.imread("Img\TEST_MUSIC_single_line_nowriting_V2.png")))
#image = ski.color.rgb2gray(ski.color.rgba2rgb(io.imread("Img/fur_elise_one_line.png")))
#image = ski.color.rgb2gray(ski.color.rgba2rgb(io.imread("Img/halfnote.png")))
#image = ski.color.rgb2gray(ski.color.rgba2rgb(io.imread("Img/fullNote_temp_oneline.png")))
image = ski.color.rgb2gray(ski.color.rgba2rgb(io.imread("Img/happybirthday.png")))


ycoord = staff_lines(image)
xcoord = bar_tail_lines(ski.util.invert(image))


cx, cy, radii = houghcircles(image, ycoord, xcoord)



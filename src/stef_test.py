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

def houghlines():
    image = ski.color.rgb2gray(ski.color.rgba2rgb(io.imread("Img\TEST_MUSIC_single_line.png")))
    thresh = threshold_otsu(image)
    binary = image > thresh

    binary = ski.util.invert(binary)

    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(binary, theta=tested_angles)

    # Generating figure 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(binary, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [
        np.rad2deg(theta[0] - angle_step),
        np.rad2deg(theta[-1] + angle_step),
        d[-1] + d_step,
        d[0] - d_step,
    ]
    ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(binary, cmap=cm.gray)
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d,min_distance=3)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2))

    plt.tight_layout()
    plt.show()

    return 0

def blob():
    image = ski.color.rgb2gray(ski.color.rgba2rgb(io.imread("Img\TEST_MUSIC_single_line_nowriting_V2.png")))
    #image = ski.color.rgb2gray(ski.color.rgba2rgb(io.imread("Img/fur_elise_one_line.png")))
    #image = ski.color.rgb2gray(ski.color.rgba2rgb(io.imread("Img/halfnote.png")))

    denoised = restoration.denoise_nl_means(image, h=0.1, fast_mode=True)


    enhanced = exposure.equalize_adapthist(denoised)

    thresh = threshold_otsu(enhanced)
    image_gray = image > thresh

    blobs_doh = blob_doh(image_gray, min_sigma = 5, max_sigma=20, threshold=0.03)

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

blob()

def houghcircles():
    image = ski.color.rgb2gray(ski.color.rgba2rgb(io.imread("Img\TEST_MUSIC_single_line_nowriting_V2.png")))
    #image = ski.color.rgb2gray(ski.color.rgba2rgb(io.imread("Img/fur_elise_one_line.png")))
    #image = ski.color.rgb2gray(ski.color.rgba2rgb(io.imread("Img/halfnote.png")))

    denoised = restoration.denoise_nl_means(image, h=0.1, fast_mode=True)


    enhanced = exposure.equalize_adapthist(denoised)
    
    
    edges = canny(enhanced, sigma=2.0, low_threshold=0.3, high_threshold=0.6)

    io.imshow(edges)
    plt.show()
    # Detect two radii
    hough_radii = np.arange(4, 5, 1, dtype=int)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,min_xdistance=12, min_ydistance=5, threshold = 0.25)

    # Draw them
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    enhanced = color.gray2rgb(enhanced)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
        enhanced[circy, circx] = (255, 0, 0)

    ax.imshow(enhanced, cmap=plt.cm.gray)
    plt.show()

    return 0

#houghcircles()

def houghelipse():
    image_rgb = ski.color.rgba2rgb(io.imread("Img\TEST_MUSIC_single_line_nowriting_V2.png"))
    image_gray = ski.color.rgb2gray(image_rgb)
    #thresh = threshold_otsu(image)
    #image = image > thresh
    #binary = ski.util.invert(image)
    #image = binary
    
    edges = canny(image_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)

    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of the histogram for minor axis lengths.
    # A higher `accuracy` value will lead to more ellipses being found, at the
    # cost of a lower precision on the minor axis length estimation.
    # A higher `threshold` will lead to less ellipses being found, filtering out those
    # with fewer edge points (as found above by the Canny detector) on their perimeter.
    result = hough_ellipse(edges, accuracy=50, threshold=100, min_size=5, max_size=100)
    result.sort(order='accumulator')

    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = (int(round(x)) for x in best[1:5])
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    image_rgb[cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    edges = color.gray2rgb(img_as_ubyte(edges))
    edges[cy, cx] = (250, 0, 0)

    fig2, (ax1, ax2) = plt.subplots(
        ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True
    )

    ax1.set_title('Original picture')
    ax1.imshow(image_rgb)

    ax2.set_title('Edge (white) and result (red)')
    ax2.imshow(edges)

    plt.show()


def hel_mult():
    image_rgb = ski.color.rgba2rgb(io.imread("Img\TEST_MUSIC_single_line_nowriting_V2.png"))
    image_gray = ski.color.rgb2gray(image_rgb)

    edges = canny(image_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)
    

    #yc, xc, a, b, orientation 
    result = hough_ellipse(edges, accuracy=50, threshold=230, min_size=3, max_size=25)
    result.sort(order='accumulator')
    
    print(len(result))
    
    #io.imshow(edges)
    #plt.show()


    image_drawn = image_rgb.copy()
    # Draw the ellipse perimeter
    biglist = result.tolist()

    for i in range(len(result)):
        test = list(biglist[i])
        cy1, cx1 = ellipse_perimeter(int(test[1]),int(test[2]),int(test[3]),int(test[4]),int(test[5]))
        cy1 = [min(x, 52) for x in cy1]
        #cx1 = [min(x, 53) for x in cx1]
        image_drawn[cy1, cx1] = (255, 0, 0) # Draw in red

    io.imshow(image_drawn)
    plt.show()



#houghelipse()
#hel_mult()
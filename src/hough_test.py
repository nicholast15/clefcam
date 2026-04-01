import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data
import skimage as ski

import matplotlib.pyplot as plt
from matplotlib import cm


# Constructing test image
#image = np.zeros((200, 200))
#idx = np.arange(25, 175)
#image[idx, idx] = 255
#image[draw_line(45, 25, 25, 175)] = 255
#image[draw_line(25, 135, 175, 155)] = 255
image = ski.io.imread("Img/blow_single.png")
image = ski.color.rgba2rgb(image)
image = ski.color.rgb2gray(image)
image = ski.util.img_as_ubyte(image)
image = ski.util.invert(image)

# Classic straight-line Hough transform
# Set a precision of 0.5 degree.
tested_angles = np.linspace(-0.01, 0.01, 2, endpoint=False)
#for i in range(0, tested_angles.size):
#    if tested_angles[i] >=0:
#        tested_angles[i] += np.pi/4
#    else:
#        tested_angles[i] -= np.pi/4
h, theta, d = hough_line(image, theta=tested_angles)

# Generating figure 1
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
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

ax[2].imshow(image, cmap=cm.gray)
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=10)):
    if dist != 0: #Check if lines are to the right of the time signature
        print(angle, dist)
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2))

plt.tight_layout()
plt.show()
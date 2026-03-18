'''
Main code containing music detection algorithms. 
Inputs: .png format images of sheet music
Outputs: ABC notation text file, eventually will play music
Stefan Py and Nicholas Trimble, 03/2026
'''
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from matplotlib import cm

'''
input: greyscale image of a single bar of the staff
output: tuple representing the Y-coordinate of each line of the staff, in pixels from the top of the image
        can also return angle input image is rotated by (degrees)
'''
def staff_lines(image):
    test_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = ski.transform.hough_line(image, theta=test_angles)

    accum, angles, dist = ski.transform.hough_line_peaks(h, theta, d, min_distance=5, num_peaks=5)
    dist = dist.astype(np.int32) #max image height of 2^32-1
    return -1*dist, np.rad2deg(np.mean(angles))
    

#for debugging
def imshow(image, isGray=True):
    if isGray:
        plt.imshow(image, cmap=cm.gray)
    else:
        plt.imshow(image)
    plt.show()

def main():
    filepath = "Img/blow_single.png" #todo: make commandline arg
    im = ski.io.imread(filepath)

    #Preprocessing
    if im.ndim == 3:            #colour image
        if im.shape[2] == 4:    #RGBA image
            im = ski.color.rgba2rgb(im)
        im = ski.color.rgb2gray(im)
    
    if im.dtype == 'float64':   #convert floats to uint8
        im = ski.util.img_as_ubyte(im)

    im = ski.util.invert(im)    #Its preferable for features to be represented by 255
    #imshow(im)

    #Staff line location
    lines, rotation = staff_lines(im)

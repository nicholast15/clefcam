'''
Main code containing music detection algorithms. 
Inputs: .png format images of sheet music
Outputs: ABC notation text file, eventually will play music
Stefan Py and Nicholas Trimble, 03/2026
'''
import numpy as np
import skimage as ski
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import cm

'''
input: greyscale image of a single bar of the staff
output: tuple representing the Y-coordinate of each line of the staff, in pixels from the top of the image
        can also return angle input image is rotated by (degrees)
'''
def staff_lines(image):
    tested_angles = np.linspace(-np.pi / 4, np.pi / 4, 180, endpoint=False)
    for i in range(0, tested_angles.size):  #check within 45deg of horizontal
        if tested_angles[i] >=0:           #refactor: make more parameterizable tested_angle function, or houghpeak function
            tested_angles[i] += np.pi/4
        else:
            tested_angles[i] -= np.pi/4
    h, theta, d = ski.transform.hough_line(image, theta=tested_angles)

    accum, angles, dist = ski.transform.hough_line_peaks(h, theta, d, min_distance=5)
    dist = dist.astype(np.int32) #max image height of 2^32-1
    return -1*dist, np.rad2deg(np.mean(angles))

'''
Simple segmentation algo, something more advanced from class could be used 
but this is easiest given the rigid formatting of the image
Input: image dimensions, staff line y coordinates
Output: Array representing the horizontal borders to slice image 
'''
def staff_borders(lines, dims):
    assert len(lines) % 5 == 0, "Detected staff lines not a multiple of 5; staff_borders will not work"
    v, h = dims

    borders = [0]       #start with top

    for i in range(4, len(lines)-5, 5):
        midpt = (lines[i] + lines[i+1]) // 2    #floor division; need whole number pixel
        borders.append(int(midpt))

    borders.append(v-1) #add bottom
    return borders

'''
Input: Image of sheet music we're processing, border array from staff_borders
Output: Array of images representing a line of bars each
'''
def staff_slice(image, borders):
    slices = []
    i = 0
    for j in range(0, len(borders)-1):
        start = borders[j]
        end = borders[j+1]-1    #dont include the start of the next slice
        slices.append(image[start:end])
        
    return slices

'''
performs singular template matching, returning score and location of best match
'''
def template_match(image, template):
    res = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED) #COEFF NORMED found best in experiments
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    return max_val, max_loc


'''
input: stripe of input image containing one line of the staff (and therefore one clef)
output: string identifying best clef match, (clef locations?)
'''
def clef_match(line):
    template_root = "Img/templates/"
    sizes = [32, 64, 128]       #template sizes to allow for diff res images
    clefs = ["GClef", "FClef"]  #clefs to check #todo: add percussion
    score = []

    #pick size to template with
    h = line.shape[0]
    if h < sizes[0]:    #verify we have a small enough template
        print(f"Error: Minimum template size {sizes[0]} exceeds height of line {h}")
        return
    else:
        size = sizes[0]     #default to smallest
        for i in range(1, len(sizes)):  #check if there is a larger template usable
            if sizes[i] > h:
                break
            size = sizes[i]
        #print(size)

        for clef in clefs:  #check all clefs at the determined size
            template_file = template_root + clef + '_' + str(size) + ".png"
            template = cv.imread(template_file, cv.IMREAD_GRAYSCALE)
            assert template is not None, "Template file " + template_file + " not found"
            max_val, max_loc = template_match(line, template)
            #print(max_val, max_loc)
            score.append(max_val)

        match_i = np.argmax(np.array(score))    #find the clef with the highest match value
        match = clefs[match_i]
        return match



#for debugging
def imshow(image, isGray=True):
    if isGray:
        plt.imshow(image, cmap=cm.gray)
    else:
        plt.imshow(image)
    plt.show()

def main(filepath):
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
    #print(lines, rotation)

    #rotate the image based on rotation
    #for now assume horizontal: below slicing will need to account for lines being horiz+vert comps

    #slice the image into stripes based on groups of 5 in lines
    borders = staff_borders(lines, im.shape)
    slices = staff_slice(im, borders)
    #for s in slices:
    #    imshow(s)

    #check the clef and signature
    clef = clef_match(ski.util.invert(slices[0])) #just check for clef on the first line
    print(clef)



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

    #filters lines to be at least 3 pix apart, and within 1% of strongest line
    accum, angles, dist = ski.transform.hough_line_peaks(h, theta, d, min_distance=3, threshold=0.99*np.max(h))
    dist = np.sort(-1*dist)
    dist = dist.astype(np.int32) #max image height of 2^32-1
    return dist, np.rad2deg(np.mean(angles))

'''
input: slice of an image with clef, time signature, sharps/flats removed
output: x coordinates of bar lines and note tails that are exactly vertical
'''
def bar_tail_lines(image):
    tested_angles = np.linspace(-0.01, 0.01, 2, endpoint=False)
    h, theta, d = ski.transform.hough_line(image, theta=tested_angles)

    accum, angles, dist = ski.transform.hough_line_peaks(h, theta, d, min_distance=10)
    dist = dist.astype(np.int32)
    return dist

'''
Simple segmentation algo, something more advanced from class could be used 
but this is easiest given the rigid formatting of the image
Input: image dimensions, staff line y coordinates
Output: Array representing the horizontal borders to slice image 
'''
def staff_borders(lines, dims, padding = 2):
    assert len(lines) % 5 == 0, "Detected staff lines not a multiple of 5; staff_borders will not work"
    v, h = dims
    borders = []

    dist = lines[2] - lines[0]  #get dist btn middle and top of staff
    distpad = dist * padding    #pad a bit to get area around staff

    for i in range(2, len(lines), 5):
        top = lines[i] - distpad
        bot = lines[i] + distpad
        borders.append(int(top))
        borders.append(int(bot))
    print(borders)
    return borders

'''
Input: Image of sheet music we're processing, border array from staff_borders
Output: Array of images representing a line of bars each
'''
def staff_slice(image, borders):
    slices = []
    for j in range(0, len(borders)-1, 2):
        start = borders[j]
        end = borders[j+1]
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
single template matching for individual symbols on the staff
input: stripe of input image containing one line of the staff
output: string identifying best match to feature, coordinate of the top right corner
'''
def templ_match(line, feature): #generalize this for more templating
    if feature == "clef":
        feats = ["GClef", "FClef"]
    elif feature == "ts":
        feats = ["4-4", "3-4", "2-2"]
    else:
        raise RuntimeError #not a valid feature to check

    template_root = "Img/templates/"
    sizes = [16, 32, 64, 128]       #template sizes to allow for diff res images
    score = []
    locns = []

    #pick size to template with
    h = line.shape[0]
    if h < sizes[0]:    #verify we have a small enough template
        print(f"Error: Minimum template size {sizes[0]} exceeds height of line {h}")
        return

    size = sizes[0]     #default to smallest
    for i in range(1, len(sizes)):  #check if there is a larger template usable
        if sizes[i] > h:
            break
        size = sizes[i]
    #print(size)

    for f in feats:  #check all features at the determined size
        template_file = template_root + f + '_' + str(size) + ".png"
        template = cv.imread(template_file, cv.IMREAD_GRAYSCALE)
        assert template is not None, "Template file " + template_file + " not found"
        max_val, max_loc = template_match(line, template)
        print(f, max_val, max_loc)
        topright = (max_loc[0]+ template.shape[1], max_loc[1]) #these indices are opposing
        score.append(max_val)
        locns.append(topright)

    match_i = np.argmax(np.array(score))    #find the clef with the highest match value
    match = feats[match_i]
    coord = locns[match_i]
    return match, coord

'''
input: image of notes in one bar line, y-coordinate of staff lines, x-coordinate of bar lines
output: xy location of notes on image
'''
def note_detection(image, ycoord, xcoord):

    height, width = image.shape

    lines = np.ones((height, width), dtype=np.uint8) * 255
    for i in range(len(ycoord)):
        for j in range(width):
            lines[ycoord[i]][j] = 0

    for i in range(len(xcoord)):
        for j in range(height):
            lines[j][xcoord[i]] = 0
            lines[j][xcoord[i]+1] = 0
            lines[j][xcoord[i]-1] = 0


    image_nolines = ski.util.invert(np.clip(ski.util.invert(image) - ski.util.invert(lines),0,255))

    denoised = ski.restoration.denoise_nl_means(image_nolines, h=0.05, fast_mode=True)
    enhanced = ski.exposure.equalize_adapthist(denoised)

    thresh = ski.filters.threshold_otsu(enhanced)
    binary = enhanced > thresh

    morph = ski.morphology.area_closing(binary,8,8)

    thresh = ski.filters.threshold_otsu(morph)
    morph = morph > thresh
    
    edges = ski.feature.canny(morph, sigma=2.0, low_threshold=0.1, high_threshold=0.5)

    hough_radii = np.arange(4, 10, 1, dtype=int)
    hough_res = ski.transform.hough_circle(edges, hough_radii)

    accums, cx, cy, radii = ski.transform.hough_circle_peaks(hough_res, hough_radii,min_xdistance=15, min_ydistance=10, threshold = 0.3)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image = ski.color.gray2rgb(image)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = ski.draw.circle_perimeter(center_y, center_x, radius, shape=image.shape)
        image[circy, circx] = (255, 0, 0)

    ax.imshow(image, cmap=plt.cm.gray)
    plt.show()

    return cx, cy, radii

'''
input: y coordinates of a note, y-coordinate of staff lines
output: ABC notation of the music
'''

def note_pos(y, staff_ycoord):
    gap = abs(staff_ycoord[0]-staff_ycoord[1])
    dist = y - staff_ycoord[0]
    pos = np.round(dist/gap*2) 
    return pos

'''
input: image of notes in one bar line, x and y coordinate of a note, x-coordinate of bar lines
output: note type
'''
def note_type(image, cx, cy, radii, vert_xcoord, ind):
    tot = 0
    count = 0
    radius = radii[ind]
    x = cx[ind]
    y = cy[ind]

    for i in range(np.floor(radius/2)*2+1):
        for j in range(np.floor(radius/2)*2+1):
            tot += image[y + i - np.floor(radius/2),x + j - np.floor(radius/2)]
            count += 1

    if tot/count > 0.5:
        return 0
    else:
        for i in range(len(vert_xcoord)):
            if vert_xcoord[i] > np.floor(x - radius*2) or vert_xcoord[i] < np.floor(x + radius*2):
                return 2
            else:
                return 1





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

    clef1, coord1 = templ_match(ski.util.invert(slices[0]), "clef") #just check for clef on the first line
    print(clef1, coord1) #coord uses cartesian, while the iamge follows image convention
    ts, coord2 = templ_match(ski.util.invert(slices[0]), "ts")
    print(ts, coord2)
    s1 = slices[0][:, coord2[0]:]
    imshow(s1)
    #key signature will be between staff and TS

    lines, rotation = staff_lines(slices[-1])
    vert_lines = bar_tail_lines(slices[-1])
    note_detection(ski.util.invert(slices[-1]),lines,vert_lines)

    for s in slices[1:]:
        #find the clef to crop - TODO: sharps and flats as well
        clef, coord = templ_match(ski.util.invert(s), "clef")
        s = s[:,coord[0]:]   #start from x location of top right match point- cut out clef
        print(bar_tail_lines(s))
        imshow(s)




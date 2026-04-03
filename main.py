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
output: string identifying best match to feature, coordinate of the top right corner and bottom left corner
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
        print(f"Candidate {f},\tsz:{size}, conf: {max_val}, loc:{max_loc}")
        topright = (max_loc[0] + template.shape[1], max_loc[1]) #these indices are opposing
        botleft  = (max_loc[0], max_loc[1] + template.shape[0])
        score.append(max_val)
        locns.append((topright, botleft))

    match_i = np.argmax(np.array(score))    #find the clef with the highest match value
    match = feats[match_i]
    coords = locns[match_i]
    return match, coords

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

    morph = ski.morphology.area_closing(binary,4,8)

    thresh = ski.filters.threshold_otsu(morph)
    morph = morph > thresh
    
    edges = ski.feature.canny(morph, sigma=2.0, low_threshold=0.1, high_threshold=0.5)

    #ski.io.imshow(morph)
    #plt.show()

    hough_radii = np.arange(3, 10, 1, dtype=int)
    hough_res = ski.transform.hough_circle(edges, hough_radii)

    accums, cx, cy, radii = ski.transform.hough_circle_peaks(hough_res, hough_radii,min_xdistance=10, min_ydistance=10, threshold = 0.3)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image = ski.color.gray2rgb(image)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = ski.draw.circle_perimeter(center_y, center_x, radius, shape=image.shape)
        image[circy, circx] = (255, 0, 0)

    #ax.imshow(image, cmap=plt.cm.gray)
    #plt.show()

    return cx, cy, radii

'''
input: y coordinates of a note, y-coordinate of staff lines
output: position by number relative to top of staff lines
'''

def note_pos(y, staff_ycoord):
    gap = abs(staff_ycoord[0]-staff_ycoord[1])
    dist = y - staff_ycoord[0]
    pos = np.round(dist/gap*2) 
    return int(pos)

'''
input: image of notes in one bar line, x and y coordinate of a note, x-coordinate of bar lines
output: note timing type
'''
def note_type(image, cx, cy, radii, staff_ycoord, vert_xcoord, ind, stack):
    radius = radii[ind]
    x = cx[ind]
    y = cy[ind]
    tot = 0

    thresh = ski.filters.threshold_otsu(image)
    image = image > thresh

    for i in range(3):
        if x - 1 >= 0:
            for j in range(3):
                if image[y-1+i, x-1+j] == True:
                    tot += 1


    if tot/9 < 0.5 or stack == True:
        temp_timing, dif = dark_note_differentiate(image, cx, cy, radii, ind, staff_ycoord, stack)

        return temp_timing, note_pos(cy[ind+dif], staff_ycoord), cx[ind+dif]
    else:
        for i in range(len(vert_xcoord)):
            if vert_xcoord[i] > x - radius*2 and vert_xcoord[i] < x + radius*2:
                return 2, note_pos(y,staff_ycoord), x
            
        return 1, note_pos(y,staff_ycoord), x


def dark_note_differentiate(image, cx, cy, radii, ind, ycoord, stack):
    x = cx[ind]
    y = cy[ind]
    radius = radii[ind]

    height, width = image.shape

    lines = np.ones((height, width), dtype=np.uint8) * 255
    for i in range(len(ycoord)):
        for j in range(width):
            lines[ycoord[i]][j] = 0

    thresh = ski.filters.threshold_otsu(image)
    image = image > thresh

    image_nolines = ski.util.invert(np.clip(ski.util.invert(image) - ski.util.invert(lines),0,255))

    low = max(0, x - radius*2)
    high = min(x + radius*5, width)
    note = image_nolines[:,low:high]

    thresh = ski.filters.threshold_otsu(note)
    note = note > thresh

    note = ski.morphology.area_opening(note,10,4)
    note = ski.morphology.area_closing(note,4,4)

    if stack == True:
        edges = ski.feature.canny(note, sigma=2.0, low_threshold=0.1, high_threshold=0.5)
        
        hough_radii = np.arange(3, 10, 1, dtype=int)
        hough_res = ski.transform.hough_circle(edges, hough_radii)

        accums, tcx, tcy, tradii = ski.transform.hough_circle_peaks(hough_res, hough_radii, threshold = 0.3, total_num_peaks = 1)

        #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        note = ski.color.gray2rgb(note.astype('uint8') * 255)
        circy, circx = ski.draw.circle_perimeter(tcy[0], tcx[0], tradii[0], shape=note.shape)
        note[circy, circx] = (255, 0, 0)
        #ax.imshow(note, cmap=plt.cm.gray)
        #plt.show()

        if abs(y-tcy[0]) < abs(cy[ind+1] - tcy[0]):
            return 8,0
        else:
            return 8,1
    else:
        stem_dir = -1
        if note_pos(y, ycoord) < 5:
            stem_dir = 1
        
        return detect_stem_end(note, x, y, radius, ycoord, stem_dir),0

def detect_stem_end(image, x, y, radius, ycoord, stem_dir):
    gap = abs(ycoord[0] - ycoord[1])

    point = int(np.round(y + stem_dir * gap * 3.5))

    thresh = ski.filters.threshold_otsu(image)
    image = image > thresh

    tot1 = 0
    tot2 = 0

    dist = x - stem_dir*radius

    height, width = image.shape

    for i in range(3):
        for j in range(3):
            if width > dist + 1 + i:
                if image[point - stem_dir * (1 + j), dist + 1 + i] == False:
                    tot1+=1
            if 0 > dist - 1 - i:
                if image[point - stem_dir * (1 + j), dist - 1 - i] == False:
                    tot2+=1
    
    if tot1/9 > 0.5 or tot2/9 > 0.5:
        return 8
    else:
        return 4




#for debugging
def imshow(image, isGray=True):
    if isGray:
        plt.imshow(image, cmap=cm.gray)
    else:
        plt.imshow(image)
    plt.show()

#def multimatch(im, templ, thres=0.75, min_dist=5):
#    #find multiple matches to a template and filter close results
#    res = cv.matchTemplate(im, templ, cv.TM_CCOEFF_NORMED)
#    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
#    print(max_val, templ.shape)
#    imshow(res)
#    matches = np.where(res >= thres)
#    coords = list(zip(matches[1], matches[0])) #swap coords to x,y
#    return coords
#    print(coords)
#
#    if not coords:
#        return [] #failed to find anything
#
#    #remove overlaps
#    coords = sorted(coords)
#    filt = [coords[0]]
#    for coord in coords[1:]:
#        if abs(coord[0] - filt[-1][0]) > min_dist:
#            filt.append(coord)
#    return filt

'''
input: limited span of a line containing key signature
outputs: number of sharps and flats detected
'''
def key_extract(im):
    thres = 0.75
    template_root = "Img/templates/"
    sizes = [8, 16, 32]       #template sizes to allow for diff res images
    sharp, flat = 0, 0

    for size in sizes:
        sharp_t = cv.imread(template_root + "sharp_" + str(size) + ".png", cv.IMREAD_GRAYSCALE)
        flat_t = cv.imread(template_root + "flat_" + str(size) + ".png", cv.IMREAD_GRAYSCALE)

        if sharp_t.shape[0] > im.shape[0] or sharp_t.shape[1] > im.shape[1]:
            break
        if flat_t.shape[0] > im.shape[0] or flat_t.shape[1] > im.shape[1]:
            break

        shval = template_match(im, sharp_t)
        flval = template_match(im, flat_t)
        #voting
        if shval > flval:
            sharp = sharp + 1
        else:
            flat = flat + 1
    
    print(sharp, flat)
    
    return "sharp" if sharp>flat else "flat"

'''
Input: section containing key signature
Output: count of number of symbols
'''
def keysig_count(im, ycoord):
    height, width = im.shape
    thres = ski.filters.threshold_otsu(im)
    im = im > thres

    #create image with the staff lines
    lines = np.ones((height, width), dtype=np.uint8) * 255
    for i in range(len(ycoord)):
        for j in range(width):
            lines[ycoord[i]][j] = 0
    #remove those lines on the image
    nolines = ski.util.invert(np.clip(ski.util.invert(im) - ski.util.invert(lines),0,255))
    imshow(nolines)

    #remove elements connected to the edge- remainders of the clef
    cleanbord = ski.segmentation.clear_border(nolines)
    imshow(cleanbord)

    #connect remainders and count
    ksz = width//10 if width//10 > 2 else 2
    kern = ski.morphology.disk(width//10)
    closed = ski.morphology.dilation(cleanbord, kern)
    imshow(closed)
    labeled = ski.measure.label(closed)
    count = len(ski.measure.regionprops(labeled))
    return count


"""
Build ABC notation header
"""
def build_abc_header(title, time_sig, n_sharps, n_flats):

    if n_sharps > 0:
        key_str = ['C', 'G', 'D', 'A', 'E', 'B', 'F#'][min(n_sharps - 1, 6)] + ' major'
    elif n_flats > 0:
        key_str = ['C', 'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb'][min(n_flats - 1, 6)] + ' major'
    else:
        key_str = 'C major'
    
    header = f"""X:1\nT:{title}\nM:{time_sig}\nL:1/8\nK:{key_str}"""
    return header

"""
Export detected music information to ABC notation file
"""
def export_to_abc(filename, clef, header):
    
    body = ""
    for bar_idx, notes_in_bar in enumerate(bars_notes):
        
        body += "| "
    
    abc_content = header + body + "\n"
    
    with open(filename, 'w') as f:
        f.write(abc_content)
    
    print(f"Exported to {filename}")

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

    #imshow(slices[0])
    clefid, clefloc = templ_match(ski.util.invert(slices[0]), "clef") #just check for clef on the first line
    print(clefid, clefloc) #coord uses cartesian, while the iamge follows image convention
    tsid, tsloc = templ_match(ski.util.invert(slices[0]), "ts")
    print(tsid, tsloc)

    #find key signature, slice out section between clef and time
    key_im = slices[0][:, clefloc[0][0]:tsloc[1][0]]
    keysig = key_extract(ski.util.invert(key_im))
    print(keysig)
    first_staff = lines[0:5] - borders[0]       #get the staff lines in this first slice, relative to its own coords
    n_keysig = keysig_count(key_im, first_staff)
    print(n_keysig)
    
    init = True #to change behaviour for the first line
    for s in slices:
        #crop out the time signature and clef, leaving only notes (and repeats which we will ignore)
        if init:
            s = s[:, tsloc[0][0]:]  #start from the right of the time signature
            init = False
        else:
            s = s[:, tsloc[1][0]:]  #start from left of the time signature (right of the key sig)
        #print(bar_tail_lines(s))
        #imshow(s)
        #continue

        #note detection
        lines, rotation = staff_lines(s)
        vert_lines = bar_tail_lines(s)
        cx, cy, radii = note_detection(ski.util.invert(s),lines,vert_lines)

        idx = np.lexsort([cy,radii,cx])
        cx = cx[idx]
        cy = cy[idx]
        radii = radii[idx]

        pos = []
        timing = []
        abs_pos_x = []
        
        for i in range(len(cx)):
            skip = False
            nogo = False

            if i != 0 and skip == False:
                if cx[i] - radii[i-1]*3 < cx[i-1] or cx[i] == cx[i-1]:
                    skip = True
                    nogo = True

            if i != len(cx)-1:
                if cx[i] + radii[i]*4 > cx[i+1] and abs(cy[i]-cy[i+1]) > radii[i]*2:
                    temp_time, sheet_pos, temp_abs_pos_x  = note_type(ski.util.invert(s), cx, cy, radii, lines, vert_lines, i, True)
                    skip = True
            
            
            if skip == False:
                temp_time, sheet_pos, temp_abs_pos_x = note_type(ski.util.invert(s), cx, cy, radii, lines, vert_lines, i, False)

            if len(abs_pos_x) != 0:
                for j in range(len(abs_pos_x)):
                    if abs_pos_x[j] < temp_abs_pos_x + 3 and abs_pos_x[j] > temp_abs_pos_x - 3:
                        nogo = True
                        break

            if nogo == False:
                timing.append(temp_time)
                pos.append(sheet_pos)
                abs_pos_x.append(temp_abs_pos_x)

        print(pos)
        print(timing)
        #print(cx)
        #print(abs_pos_x)
        #break


im = main("Img/blow.png")

def test():
    cases = ["blow.png", 
             "sax.jpg", 
             "cradle.png", 
             "teehans.png",
             "bass.png",
             "eflat.png"
    ] #fur elise is in 3/8, not supported. also too complicated
    for c in cases:
        main("Img/" + c)




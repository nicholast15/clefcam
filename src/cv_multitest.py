#testing OpenCV methods, especially template matching for staff/time signature recognition
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('Img/blow.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
img2 = img.copy()
template = cv.imread('Img/templates/GClef_med.png', cv.IMREAD_GRAYSCALE)
assert template is not None, "file could not be read, check with os.path.exists()"
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR',     #TM_CCORR* and TM_SQDIFF* all failed with a single bar
            'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']



img = img2.copy()
res = cv.matchTemplate(img,template,cv.TM_CCOEFF_NORMED)
threshold = np.percentile(res, 99.995)
print(threshold)
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), 0, 2)

plt.subplot(121),plt.imshow(res,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle("ccoeff_normed")

plt.show()
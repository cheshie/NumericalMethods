
# I just tried to experiemnt with how a matrix should look like
# to be able to convert it to image

# Here is simple 15 x 15 matrix
x = [[  0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255],
 [255, 255, 0, 0, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 0, 255, 0, 255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255],
 [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [  0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0],
 [  0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [  0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [  0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

from matplotlib import pyplot as plt
from numpy import array, dstack
# By using it three times we can have rgb matrix, but matplotlib does not like it
# Only with dstack (# "Stack arrays in sequence depth wise (along third axis).")
plt.imshow(dstack([x,x,x]), interpolation='nearest')
plt.show()
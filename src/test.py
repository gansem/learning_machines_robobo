import cv2
import numpy as np

img = np.load('test_img.npy')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

low = np.array([0, 50, 20])
high = np.array([5, 255, 255])
mask = cv2.inRange(hsv, low, high)

left = mask[:, 0:25]
mid_left = mask[:, 25:51]
mid = mask[:, 51:77]
mid_right = mask[:, 77:103]
right = mask[:, 103:]

cam_values = [left, mid_left, mid, mid_right, right]

cam_values = [np.sum(value)/(value.shape[0] * value.shape[1]) for value in cam_values]

cv2.imwrite('a.png', mask)
0
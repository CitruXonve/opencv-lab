from __future__ import print_function
import cv2
import struct

print('Specify input image:', end='')

try:
	filename = raw_input()
except:
	filename = input()

origin = cv2.imread(filename, cv2.IMREAD_COLOR)
b, g, r = cv2.split(origin)
origin = cv2.merge([r, g, b])

width, height, channel = origin.shape

for i in range(width):
    for j in range(height):
            r = (origin[i,j][0] >> 3) & 0x1F
            g = (origin[i,j][1] >> 2) & 0x3F
            b = (origin[i,j][2] >> 3) & 0x1F
            tmp = (r<<11)+(g<<5)+b
            print('0x%04X,' % tmp,end=' ')
            # print(origin[i,j],end=' ')
            # print(struct.pack('H', tmp),end=' ')
    print()

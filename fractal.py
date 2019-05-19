import cv2
import numpy as np
import math

# 먼저 어떤 맵의 픽셀 좌표를 복소평면상에서 주어진 좌표로 대응시키는 방법이 필요하다.
# 그렇다면 인자로 배열과, 배열의 가로세로길이=(mw,mh), x,y,w,h를 받아 이를 대응시킨다.
# 물론 w,h는, w는 실수 측 크기, h는 복소수 측 크기이다.


def convertPoint(mx, my, mw, mh, x, y, w, h):
    nx = x+mx*w/mw
    ny = y+h-my*h/mh
    return nx, ny


def checkDiv(x, y, cx, cy, n):
    x_ = 0
    y_ = 0
    for i in range(n):
        if x*x+y*y > 4:
            return i
        x_ = x*x-y*y+cx
        y_ = 2*x*y+cy
        x = x_
        y = y_
    return 0


def checkMandelbrotDiv(x, y, n):
    return checkDiv(x, y, x, y, n)


def getMandelbrotArea(x, y, w, h, n, r=300):
    mw, mh = int(w*r), int(h*r)
    image = np.ndarray([mh, mw], np.float32)
    for i in range(0, mh):
        for j in range(0, mw):
            x_, y_ = convertPoint(j, i, mw, mh, x, y, w, h)
            # image[i, j] = 255-checkMandelbrotDiv(x_, y_, n)*255/10
            if checkMandelbrotDiv(x_, y_, n) > 0:
                image[i, j] = 0
            else:
                image[i, j] = 255
    return np.array(image, np.uint8)


level = 500
maxHeight = 3
heightRate = 0.95
imageHeight = 100
whRate = 3/2

fps = 20.0
width = int(imageHeight*whRate)
height = imageHeight
fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')

out = cv2.VideoWriter('video.avi', fcc, fps, (width, height))

for i in range(0, level+1):
    print(i, int(i*100/level))
    h = maxHeight*pow(heightRate, i)
    r = imageHeight/h
    image = getMandelbrotArea(-1.6, -h/2, h*whRate, h,
                              int(math.log((maxHeight/h))*2.4+14), r)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    out.write(image)

out.release()
cv2.destroyAllWindows()
cv2.waitKey(0)

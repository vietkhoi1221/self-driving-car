import cv2
import numpy as np
from scipy.stats import itemfreq
from picamera.array import PiRGBArray
from picamera import PiCamera
import pickle
import time
import dc
def get_dominant_color(image, n_colors):
    pixels = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    flags, labels, centroids = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    return palette[np.argmax(itemfreq(labels)[:, -1])]

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


l=0
bins=8
model = pickle.load(open("kh.sav","rb"))
camera = PiCamera()
camera.resolution = (1280, 784)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(1280,784))
time.sleep(0.1)
dc.dong_co_chay_tien(20)
k = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = frame.array
    frame = frame[:300,950:1280]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray, 37)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                              1, 50, param1=120, param2=40)
    
    if not circles is None:
        circles = np.uint16(np.around(circles))
        max_r, max_i = 0, 0
        for i in range(len(circles[:, :, 2][0])):
            if circles[:, :, 2][0][i] > 50 and circles[:, :, 2][0][i] > max_r:
                max_i = i
                max_r = circles[:, :, 2][0][i]
        x, y, r = circles[:, :, :][0][max_i]
        if y > r and x > r:
            square = frame[y-r:y+r, x-r:x+r]
            dominant_color = get_dominant_color(square, 2)
            
            if dominant_color[2] > 100:
                sign = cv2.resize(square,(200,200))
                fv_hu_moments = fd_hu_moments(sign)
                fv_histogram  = fd_histogram(sign)
                global_feature = np.hstack([fv_histogram, fv_hu_moments])
                prediction2 = model.predict(global_feature.reshape(1,-1))
                if(prediction2==1):
                    print("Dung")
                    if k ==0:
                        k=1
                        a = time.time()
                        dc.dong_co_dung()
                        while True:
                            if (time.time()- a >5):
                                break
                        dc.dong_co_chay_tien(25)
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #if (time.time()-a>30):
            #k = 1
    cv2.imshow('camera', frame)
    rawCapture.truncate(0)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        dc.dong_co_dung()
        break

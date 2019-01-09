import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import dk
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
    left_fit = []
    left_line = np.array([])
    right_line = np.array([])
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)   
    right_fit_average = np.average(right_fit, axis=0)
    if(len(left_fit_average.shape)!=0):
        left_line = make_coordinates(image, left_fit_average)
    if(len(right_fit_average.shape)!=0):    
        right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if(line.shape[0]!=0):
                x1,y1,x2,y2 = line.reshape(4)
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(100,height),(1100,height),(680,500)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

camera = PiCamera()
camera.resolution = (1280, 784)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(1280,784))
bins=8
direction = "straight"
dk.servo(90)
dk.dong_co_chay_tien(20)
# sleep 0.1 second
time.sleep(0.1)
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    try:
        dau = []
        cuoi = []
        dodai = []
        k=0
        goc = 0
        image = frame.array
        lane = np.copy(image)
        blur = cv2.GaussianBlur(lane,(5,5),0)
        hsv = cv2.cvtColor(lane, cv2.COLOR_BGR2HSV)
        low_yellow = np.array([18,94,140])
        up_yellow = np.array([48,255,255])
        mask = cv2.inRange(hsv, low_yellow, up_yellow)
        canny = cv2.Canny(mask, 50,150)
        cropped_image = region_of_interest(canny)
        lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]), minLineLength=40, maxLineGap=5)
        if lines is not None:
            averaged_lines = average_slope_intercept(lane, lines)
            print(averaged_lines)
            for line in averaged_lines:
                if(line.shape[0]!=0):
                    x1, y1, x2, y2 = tuple(line)
                    dau.append(list([int(x2),int(y2)]))
                    cuoi.append(list([int(x1),int(y1)]))
                    # cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
                    print("Line thứ {}: x1={} x2={} y1={} y2={}".format(k,x1,x2,y1,y2))
                    k = k+ 1
                    dodai.append(np.sqrt((x1/37.8-x2/37.8)**2 +(y1/37.8-y2/37.8)**2))
            if len(dodai)>0:
                l = dodai.index(max(dodai))
                ke = np.abs(cuoi[l][1]/37.8-dau[l][1]/37.8)
                #print(dodai[l])
                #print(l)
                #print(ke)
                if(dau[l][0] < cuoi[l][0] and dau[l][1] < cuoi[l][1]):
                    goc = 90 - np.arccos(ke/dodai[l])*180/3.14
                else:
                    goc = np.arccos(ke/dodai[l])*180/3.14 
                if(dau[l][0] == cuoi[l][0]):
                    #print("di thang")
                    direction = "straight"
                else:
                    if(dau[l][1] > cuoi[l][1]):
                        if(dau[l][0] <cuoi[l][0]):
                            #print("re phai")
                            direction = "right"
                            goc -= 30
                        else:
                            #print("re trai")
                            direction = "left"
                            goc = 90 - goc
                            if (goc>55):
                                goc -=20
                            else:
                                goc -= 25
                    elif (dau[l][1] < cuoi[l][1]):
                        if(dau[l][0] >cuoi[l][0]):
                            #print("re phai")
                            direction = "right"
                            goc -= 30
                        else:
                            #print("re trai")
                            direction = "left"
                            goc = 90 - goc
                            if (goc>55):
                                goc -= 20
                            else:
                                goc -= 25
                print("goc: " +str(goc))
            line_image = display_lines(lane,averaged_lines)
            combo_image = cv2.addWeighted(lane,0.8,line_image,1,1)
            cv2.imshow("dadssd",combo_image)
        else:
            cv2.imshow("dadssd",lane)
            cv2.imshow("ddasd",mask)
        if ((goc < 15) and (goc > -15)):
            dk.dong_co_chay_tien(20)
            print("di thang")
            dk.servo(90)
        elif (direction == "left"):
            print("re trai")
            dk.servo(115)
            dk.dong_co_chay_tien(30)
            time.sleep(1.5)
            pass
        elif (direction == "right"):
            print("re phai")
            dk.servo(70)
            dk.dong_co_chay_tien(30)
            time.sleep(1)
            pass
        rawCapture.truncate(0)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            dk.dong_co_dung()
            dk.servo(90)
            #GPIO.cleanup()
            break
    except OverflowError as e:
        #GPIO.cleanup()
        print("Xay ra loi")
        

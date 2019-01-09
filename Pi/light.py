from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import pickle
import dc
model = pickle.load(open("finalized_model.sav","rb"))
labels = ["den_do","den_xanh"]

def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (1280, 780)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(1280,780))
bins=8
# sleep 0.1 second
time.sleep(0.1)
dc.dong_co_chay_tien(0)
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # convert frame to array
    a = time.time()
    image = frame.array
    # display a frame
    img = cv2.resize(image,(600,600))   
    #Chuyển từ RGB sang HSV
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #Khai báo khoảng giá trị RGB của màu đỏ
    red_lower=np.array([136,87,111],np.uint8)
    red_upper=np.array([180,255,255],np.uint8)

    #Khai báo khoảng giá trị RGB của màu xanh
    green_lower=np.array([66, 122, 129],np.uint8)
    green_upper=np.array([86,255,255],np.uint8)

    #Tìm ngưỡng màu đỏ và xanh
    red=cv2.inRange(hsv, red_lower, red_upper)
    green=cv2.inRange(hsv,green_lower,green_upper)

    #thực hiện Morphological transformation và tiến hành and ảnh đầu vào với mặt nạ mới biến đổi	
    kernal = np.ones((5 ,5), "uint8")
    red=cv2.dilate(red, kernal)
    res=cv2.bitwise_and(img, img, mask = red)
    green=cv2.dilate(green,kernal)
    res2=cv2.bitwise_and(img, img, mask = green)    

    #Xác định đường bao có màu đỏ
    im2,contours,hierarchy=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        #tính toán diện tích đường bao, nếu > 300 thì tiến hành xử lý
        area = cv2.contourArea(contour)
        if(area>300):
            #Trích xuất giới hạn của đường bao và khoanh vùng 
            x,y,w,h = cv2.boundingRect(contour)
            im1 = img[y:y+h,x:x+w]
            im1= cv2.resize(im1,(25,25))
            prediction1 = model.predict(fd_histogram(im1).reshape(1,-1))[0]
            if(prediction1 == 0):
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(img,"Den do",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))
                #print(model.predict_proba(fd_histogram(im1).reshape(1,-1)))
                #print('Đèn đỏ bay ôi')
                dc.dong_co_dung()
    #Xác định đường bao có màu xanh    
    (im2,contours,hierarchy)=cv2.findContours(green,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        #tính toán diện tích đường bao, nếu > 300 thì tiến hành xử lý
        area = cv2.contourArea(contour)
        if(area>300):
 
            #Trích xuất giới hạn của đường bao và khoanh vùng 
            x,y,w,h = cv2.boundingRect(contour)
            im2 = img[y:y+h,x:x+w]
            im2= cv2.resize(im2,(25,25))
            prediction2 = model.predict(fd_histogram(im2).reshape(1,-1))[0]
            if(prediction2 == 1):
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(img,"Den xanh",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))
                #print(model.predict_proba(fd_histogram(im2).reshape(1,-1)))
                #print('Đen xanh được phép chạy')
                dc.dong_co_chay_tien(30)
    cv2.imshow("Frame", img)
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    print(time.time()-a)
    rawCapture.truncate(0)
    #wait for 'q' key was pressed and break from the loop
    if key == ord('q'):
        dc.dong_co_dung()
        break

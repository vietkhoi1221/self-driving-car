import RPi.GPIO as GPIO          
from time import sleep

in1 = 24
in2 = 23
in3 = 12
in4 = 16
en1 = 25
en2 = 20
servoPIN = 17
temp1=1

GPIO.setmode(GPIO.BCM)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(en1,GPIO.OUT)
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
GPIO.setup(en2,GPIO.OUT)
GPIO.output(in3,GPIO.LOW)
GPIO.output(in4,GPIO.LOW)
GPIO.setup(servoPIN, GPIO.OUT)
GPIO.setwarnings(False)
p = GPIO.PWM(servoPIN, 50) # GPIO 17 for PWM with 50Hz
p.start(7.5) # Initialization
p1=GPIO.PWM(en1,1000)
p1.start(25)
p2=GPIO.PWM(en2,1000)
p2.start(25) 

def dong_co_chay_tien(speed):
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.HIGH)
    GPIO.output(in4,GPIO.LOW)
    p1.ChangeDutyCycle(speed-0)
    p2.ChangeDutyCycle(speed)

def dong_co_dung():
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW) 
    GPIO.output(in4,GPIO.LOW)

def dong_co_chay_lui(speed):
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.HIGH)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.HIGH)
    p1.ChangeDutyCycle(speed-0)
    p2.ChangeDutyCycle(speed)

def servo(goc):
    cycle = 2.5 + 10/180*goc
    p.ChangeDutyCycle(cycle)


while(1):
    try:

        x=input()
        
        if x=='r':
            print("chay tien")
            dong_co_chay_tien(30)
            x='n'
        elif x == 'l':
            print("chay lui")
            dong_co_chay_lui(30)
            x='n'
        elif x == 's':
            print('dung')
            dong_co_dung()
            x= 'n'
        elif x=='e':
            GPIO.cleanup()
            break
        else:
            a = float(x)
            cycle = 2.5 + 10/180*a
            p.ChangeDutyCycle(cycle)
            sleep(1)
    except:
        dong_co_dung()
        GPIO.cleanup()  

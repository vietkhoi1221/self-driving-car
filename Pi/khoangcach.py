import RPi.GPIO as GPIO
import time
import dc
TRIG_PIN = 14
ECHO_PIN = 15

GPIO.setmode(GPIO.BCM)
time.sleep(2)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)
batdaudo = 0
ketthucdo = 0
thoigiando = 0
khoangcach = 0
dc.dong_co_chay_tien(20)
def dokhoangcach(chodoi):
	print('bat dau do')
	time.sleep(chodoi)
	#moi giay do 1 lan
	while True:
		#kich hoat cam bien theo muc LOW-HIGH-LOW
		GPIO.output(TRIG_PIN, False)
		#cho 200 ms
		time.sleep(0.2)
		GPIO.output(TRIG_PIN, True)
		#cho 10 micro giay
		time.sleep(0.00001)
		GPIO.output(TRIG_PIN, False)
		
		#chan ECHO duoc keo xuong muc 0 cho toi khi nhan duoc tin hieu phan hoi
    		#ham time() de lay thoi gian hien tai
		batdaudo = time.time()
		while GPIO.input(ECHO_PIN) == 0:
			batdaudo = time.time()
		#co tin hieu phan hoi chan ECHO duoc keo len muc 1, 
		#wait cho den khi nhan duoc het tin hieu phan hoi
		ketthucdo = time.time()
		while GPIO.input(ECHO_PIN) == 1:
			ketthucdo = time.time()
		
		#hoan tat nhan tin hieu phan hoi, tinh khoang thoi gian phan hoi
		thoigiando = ketthucdo - batdaudo

		#van toc sieu am la 344m/s = 34400cm/s
		khoangcach = (thoigiando * 34400)/2
		print("khoang cach:"+ str(khoangcach))
		if (khoangcach<20):
			dc.dong_co_dung()
		else: 
			dc.dong_co_chay_tien(25)
dokhoangcach(1)

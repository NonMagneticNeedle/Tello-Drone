import cv2
import numpy as np
import newtello
import time
import threading

def slip_thread(x,y):
	#time.sleep(1)

	dx,dy =x-720/2,y-960/2
	if abs(dy)<960:
		if dx>0:
			#my_drone.right(20)
			#my_drone.flip('r')
			print("right")
		if dx<0:
			#my_drone.left(20)
			#my_drone.flip('l')
			print("left")
	else:
		if dy>0:
			#my_drone.up(30)
			print("up")
		if dy<0:
			#my_drone.down(30)
			print("down")
			



font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
#my_drone = newtello.Tello()
#my_drone.streamon()
#my_drone.takeoff()
#my_drone.up(70)

#my_drone.cw(360)
#my_drone.ccw(360)
"""

	redimage = np.array(image[:,:,2])

	
	red = cv2.inRange(redimage,230,255)
	mask = cv2.inRange(image[:,:,2],0,100)
	red= cv2.bitwise_and(red,red, mask=mask)
"""

t=-1

while 1:
	ret, image =cap.read()
	#ret, image =my_drone.get_snap()#(720, 960, 3)
	image = cv2.resize(image, (720, 960))
	image=cv2.flip(image,1)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_blue = np.array([144, 153,153])
	upper_blue = np.array([179, 255, 255])
	mask = cv2.inRange(hsv, lower_blue, upper_blue)
	percent  = np.std(mask)
	if percent>8 and t<=0:

		cv2.putText(image,"attack",(20,50), font, 1,(255,100,255),1,cv2.LINE_AA)
		t=20
		#slip(cX,cY)
		action_thread = threading.Thread(target=slip_thread,args=(cX,cY),daemon=True)
		action_thread.start()        
	if t>0:
		t-=1
	if t==0:
		action_thread.join()	
	M = cv2.moments(mask)
	cX = int(M["m10"] / (M["m00"]+0.0001))
	cY = int(M["m01"] / (M["m00"]+0.0001))		
	cv2.circle(image, (cX, cY), 5, (255, 0, 100), -1)		
	cv2.imshow("obj", mask)
	cv2.putText(image,str(percent),(10,30), font, 1,(255,255,255),2,cv2.LINE_AA)
	cv2.imshow("live ", image)
	
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break     
my_drone.land()
        
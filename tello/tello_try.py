import newtello
import cv2
import time
my_drone = newtello.Tello()
my_drone.streamon()
time.sleep(1)
my_drone.set_speed(20)
my_drone.get_vid()

my_drone.takeoff()


my_drone.flip('l')

my_drone.cw(30)
my_drone.cw(30)
my_drone.cw(30)
my_drone.cw(30)

my_drone.cw(30)
my_drone.cw(30)
my_drone.cw(30)

my_drone.cw(30)
my_drone.cw(30)
my_drone.cw(30)

my_drone.cw(30)
my_drone.cw(30)
my_drone.cw(30)

my_drone.flip('r')
my_drone.down(20)
time.sleep(4)
my_drone.down(20)

for _ in range(5):
	my_drone.up(30)
	my_drone.down(30)
my_drone.land()
"""
frame_rate = 1
prev = time.time()

while 1:

    ret, frame = my_drone.get_snap()
    if (time.time() - prev) <1./frame_rate:
    	continue
    prev = time.time()

    print("got")
    cv2.imshow('snap', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
    	break
"""
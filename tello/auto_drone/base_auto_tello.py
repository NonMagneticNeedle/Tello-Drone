import cv2
#activate myWincv conda env for this
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt
import numpy as np
import newtello

import time
import threading
print('Loading model...')
model_path = "nyu.h5"
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

# Load model into GPU / CPU
model = load_model(model_path, custom_objects=custom_objects, compile=False)
model.summary()
print('\nModel loaded ({0}).'.format(model_path))


def center_crop(im,h,w):
	sh,sw = im.shape[0],im.shape[1]
	y1 = sh//2-h//2
	x1 = sw//2-w//2
	roi = im[y1:y1+h, x1:x1+w]
	return roi


def depth(img):
	input1 = np.clip(np.asarray(img, dtype=float) / 255, 0, 1)
	# Compute results
	input2 = np.clip(np.asarray(np.flip(img,axis=1), dtype=float) / 255, 0, 1)
	#op = predict(model, np.array([input1,input2]))
	#output1,output2 = op[0],op[1]
	output1 = predict(model, input1)
	output2 = predict(model, input2)
	#cv2_imshow(outputs[0,:,:,0])
	rescaled1 =output1[0,:,:,0]
	rescaled1 = rescaled1 - np.min(rescaled1)
	rescaled1 = rescaled1 / np.max(rescaled1)
	rescaled1 =  cv2.resize(rescaled1, (img.shape[1],img.shape[0]))

	rescaled2  =np.flip(output2[0,:,:,0],axis=1)
	rescaled2 = rescaled2 - np.min(rescaled2)
	rescaled2 = rescaled2 / np.max(rescaled2)
	rescaled2 =  cv2.resize(rescaled2, (img.shape[1],img.shape[0]))
	rescaled=(rescaled1+rescaled2)/2
	return rescaled
def q_depth(img):
	input1 = np.clip(np.asarray(img, dtype=float) / 255, 0, 1)
	output1 = predict(model, input1)
	rescaled1 =output1[0,:,:,0]
	rescaled1 = rescaled1 - np.min(rescaled1)
	rescaled1 = rescaled1 / np.max(rescaled1)
	rescaled1 =  cv2.resize(rescaled1, (img.shape[1],img.shape[0]))
	return rescaled1




def slip_thread(center,left,right):
	#time.sleep(1)

	if np.sum(center)==0:
		print("going forward")
		my_drone.forward(20)
	else :
		if np.sum(left)<=np.sum(right):
			print("left")
			#my_drone.cw(20)
			my_drone.right(20)
		elif np.sum(left)>np.sum(right):
			print("right")
			#my_drone.ccw(20)
			my_drone.left(20)
		else:
			print("stuck")	
			#my_drone.cw(5)
my_drone = newtello.Tello()
my_drone.streamon()

def startup():

	# Custom object needed for inference and training
	my_drone.takeoff()
	my_drone.down(20)
	#my_drone.left(20)
	#my_drone.right(20)
	#my_drone.cw(30)
	#my_drone.ccw(30)
	my_drone.down(50)
	#my_drone.down(40)

start_thread = threading.Thread(target=startup,args=(),daemon=True)
start_thread.start()    



#cap = cv2.VideoCapture(0)


#my_drone.cw(360)
#my_drone.ccw(360)

frame_rate = 0.5# to set fps
prev = time.time()
scale = 6
while 1:
	#ret, img =cap.read()
	ret, img =my_drone.get_snap()#(720, 960, 3)
	if (time.time() - prev) <1./frame_rate:
		k = cv2.waitKey(1) & 0xFF
		if k == 27:
			break     
		continue
	prev = time.time()

	#img = center_crop(img,480,640)
	img_r = cv2.resize(img, (64*scale,48*scale))
	op=q_depth(img_r)
	mask = cv2.inRange(op*255,0,25)
	#cv2.imshow("img",img)
	cv2.imshow("depth",op)	
	center = center_crop(mask,48,80)
	visor = center_crop(mask,48,64*scale)
	#cv2.imshow("visor",visor)
	
	left = visor[0:48,64*(scale//2):64*scale]
	right = visor[0:48,0:64*(scale//2)]
	action_thread = threading.Thread(target=slip_thread,args=(center,left,right),daemon=True)
	action_thread.start()    
	key = cv2.waitKey(1)
	if key == 27:
		break
my_drone.land()    
cv2.destroyAllWindows()		 

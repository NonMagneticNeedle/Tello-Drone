import tensorflow as tf
import cv2
from scipy.ndimage.filters import gaussian_filter
from time import clock
#simport time.process_time as clock
import numpy as np
import newtello
import time

tf.compat.v1.disable_eager_execution()
print(tf.__version__)
############################### tuning #############################
#accuracy boosting variables
mul=1 # 2 for more accuracy but less speed
steps=1 # 1 for more accuracy but less speed , 4 for more post processing speed but less accuracy
visual =True #False; visualising takes post processing time
#sensistivity of mousec 0-1-2
kx=0.5 
ky=1
###############################################################################
KERAS_MODEL_FILE=r"C:\\Users\\Dell\\Desktop\\mytello\\hand_model.h5"
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())


# Or loading from prepared weigths file

model=tf.keras.models.load_model(KERAS_MODEL_FILE)
print('LOADED!')
def int_mean(arr):
    add = 0
    count = 0
    for i in range(arr.shape[0]):
        if arr[i]>-1:
            add=add+arr[i]
            count+=1
    if count>4:        
        return int(add/count)
    else:
        return None    
    
def find_peaks(heatmap_avg, thre=0.1, sigma=3):
    all_peaks = []
    peak_counter = 0

    for part in range(0, heatmap_avg.shape[-1],steps):
    #for part in [8,12]:
    
        map_ori = heatmap_avg[:,:,part]
        map = gaussian_filter(map_ori, sigma=sigma)


        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]

        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > thre))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
        
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        peaks_with_score_1=[]
        for k in peaks_with_score:
            if k[2]>0.2:
                peaks_with_score_1.append(k+(part,))
        #id = range(peak_counter, peak_counter + len(peaks))
        #peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        #all_peaks.append(peaks_with_score_and_id)
        all_peaks.append(peaks_with_score_1)
        peak_counter += len(peaks)
    return all_peaks, peak_counter


def parts(all_peaks):
    partx = np.ones(22)*-1
    party = np.ones(22)*-1
    for i in range(len(all_peaks)):
        for j in all_peaks[i]:
            partx[j[3]] = j[0]
            party[j[3]] = j[1]
    return  partx,party      
def process_image(model, image_orig, scale_mul=1, peaks_th=0.1, sigma=3, mode='heatmap'):
    
    scale = 368/image_orig.shape[1]
    scale = scale*scale_mul
    image =  cv2.resize(image_orig, (0,0), fx=scale, fy=scale) 

    start = clock()
    net_out = model.predict(np.expand_dims( image /256 -0.5 ,0))
    stop = clock()
    took = stop-start
    
    out = cv2.resize( net_out[0], (image_orig.shape[1], image_orig.shape[0]) )
    image_out = image_orig
    #print("image shape",(image_orig.shape[1], image_orig.shape[0]))
    center_y,center_x=None,None
    
    mask = np.zeros_like(image_out).astype(np.float32)
    if mode == 'heatmap':
        #0->22-2
        for chn in range(0, out.shape[-1]-2):
            m = np.repeat(out[:,:,chn:chn+1],3, axis=2)
            m = 255*( np.abs(m)>0.2)
            
            mask = mask + m*(mask==0)
        mask = np.clip(mask, 0, 255)
        image_out = image_out*0.8 + mask*0.2
    else:
        peaksR = find_peaks(out, peaks_th, sigma=sigma)[0]
        #print("-"*10)
        #print(peaksR)    
        #print(peaksR,len(peaksR))
        #print(parts(peaksR))
        px,py=parts(peaksR)
        #peaksL = find_peaks(-out, peaks_th, sigma=sigma)[0]
        peak1=[]
        peak2=[]
        
        for peak in peaksR:

            if(len(peak)):
                peak = peak[0]
                #cv2.drawMarker(image_out, (peak[0], peak[1]), (0,0,255), cv2.MARKER_STAR )
                peak1.append(peak[0])
                peak2.append(peak[1])
        if(len(peak1)):    
            peak1=np.array(peak1)
            peak2=np.array(peak2)        
            center_x=np.sum(peak1)//len(peak1)
            center_y=np.sum(peak2)//len(peak2)

            #cv2.drawMarker(image_out, (int(center_x),int(center_y)), (255,0,0), cv2.MARKER_STAR )
        """
        for peak in peaksL:
            if(len(peak)):
                peak = peak[0]
                cv2.drawMarker(image_out, (peak[0], peak[1]), (255,0,0), cv2.MARKER_STAR )
        """        
                
    image_out = np.clip(image_out, 0, 255).astype(np.uint8)
                
    return image_out, took,center_x,center_y,px,py
def move(x,y,img):
    xcen,ycen = img.shape[0]//2,img.shape[1]//2
    delx = x-xcen
    dely = y - ycen
    print(delx,dely)
    if delx>0:
        my_drone.ccw(5)
        print('right')
    if delx<0:
        my_drone.cw(5)
        print('left')
    if dely>0:
        #my_drone.down(20)
        print('down')
    if dely<0:
        #my_drone.up(20)
        print('up')
        




x=0
y=0
#cap = cv2.VideoCapture(0)
my_drone = newtello.Tello()
my_drone.streamon()
my_drone.takeoff()

my_drone.cw(360)
my_drone.ccw(360)

frame_rate = 0.5# to set fps
prev = time.time()

font = cv2.FONT_HERSHEY_SIMPLEX
#while 1:
for _ in range(10000):
    #ret, image =cap.read()
    ret, image =my_drone.get_snap()
    if (time.time() - prev) <1./frame_rate:
        image = cv2.resize(image, (300, 300))

        #cv2.putText(image,'bat: {}, temp: {}C, tof:{}'.format( my_drone.get_battery(),my_drone.get_temp(),my_drone.get_tof()),(10,30), font,0.5,(255,255,255),1,cv2.LINE_AA)
        
        #cv2.imshow("live", image)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break     
        continue
    prev = time.time()


    #image = cv2.resize(image, (480, 640))
    image=cv2.flip(image,1)

    image_orig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    start = clock()
    image_out, inference_took,X,Y,xs,ys = process_image(model, image,mode='none')
    frameWidth,frameHeight,channel=np.shape(image_out)
    print(xs,ys)

    X,Y = int_mean(xs),int_mean(ys)
    if visual:
        if X is not None:
            image_out = cv2.circle(image_out,(X,Y),30,(100,100, 100), 2)

            move(X,Y,image_out)
        stop = clock()
        took = stop-start
        cv2.putText(image_out,'Inference: {}s, post: {}s'.format(  np.round(inference_took,3) , np.round(took-inference_took,3) ),(10,30), font, 1,(255,255,255),2,cv2.LINE_AA)
        
        cv2.imshow("hand tracking", image_out)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
my_drone.land()




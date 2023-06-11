from keras.models import load_model #to load .h5 file
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classify=cv2.CascadeClassifier(r'C:\Users\hayay\Downloads\gp\keras\Facial-Expressions-Recognition-master\Facial-Expressions-Recognition-master\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\hayay\Downloads\gp\keras\Facial-Expressions-Recognition-master\Facial-Expressions-Recognition-master\Emotion_little_vgg.h5')

class_labels = ['Angry','happy','neutral','sad','surprise']

cap = cv2.VideoCapture(0)
#while the cam istrue
while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    #labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classify.detectMultiScale(gray,1.3,5) #scale down our frame reduce num of features to (30%) minimum neightbours

    #in the face 1)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,150,0),2) #(p1,p2),position,color=blue,thikness عرضه 
        roi_gray = gray[y:y+h,x:x+w] #regin of interest =do grayscalling for this regin 
        roi_gray =cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        #resize the particular regin 48*48 put the testing size  ??interpolation=cv2.INTER_AREA??
    # rect,face,image = face_detector(frame)
        if np.sum([roi_gray])!=0: #if there is the face 
            roi = roi_gray.astype('float')/255.0 #?? to reduce the pixel size / into 255 because it is the max is 255  to be easy to recognation
            roi = img_to_array(roi) #for mathimatical caculations
            roi = np.expand_dims(roi,axis=0) #
         # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0] #want first predect 
            label=class_labels[preds.argmax()] #the max predect og the expression if it confuse
            label_position = (x,y) #put the label in the rectangle upper side
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(50,50,255),3) #put the predict expresion in the position with green color,the size of the font is 3
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(50,50,255),3) #if no face put no face in the label
    cv2.imshow('Emotion Detector',frame)#show the fram ,name it "" 
    if cv2.waitKey(1) & 0xFF == ord('q'): #if press q the frame will stop to dont hang the cam
        break

cap.release() 
cv2.destroyAllWindows()
                            
    


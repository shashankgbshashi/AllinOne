# import keras
# from keras.models import Sequential
# from keras.layers import Dense,Convolution2D,Flatten,MaxPooling2D
#
# classifier = Sequential()
#
# classifier.add(Convolution2D(32,3,3,activation='relu',input_shape=(64,64,3)))
# classifier.add(MaxPooling2D(pool_size=(2,2)))
# classifier.add(Convolution2D(32,3,3,activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2,2)))
#
# classifier.add(Flatten())
#
# classifier.add(Dense(output_dim=128,activation='relu'))
# classifier.add(Dense(output_dim=128,activation='relu'))
# #classifier.add(Dense(output_dim=128,activation='relu'))
# classifier.add(Dense(output_dim=3,activation='sigmoid'))
#
# classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#
# from keras.preprocessing.image import ImageDataGenerator
#
# train_data = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)
#
# test_data = ImageDataGenerator(rescale=1./255)
#
# train_set = train_data.flow_from_directory(
#         'dataset/train',
#         target_size=(64, 64),
#         batch_size=2,
#         class_mode='categorical')
#
# test_set = test_data.flow_from_directory(
#         'dataset/test',
#         target_size=(64, 64),
#         batch_size=2,
#         class_mode='categorical')
#
# classifier.fit_generator(
#         train_set,
#         steps_per_epoch=51,
#         epochs=10,
#         validation_data=test_set,
#         validation_steps=14)
#
#
# import numpy as np
# from keras.preprocessing import image
#
# test_img = image.load_img('dataset/prediction/loo.jpg',target_size=(64,64))
# test_img = image.img_to_array(test_img)
# test_img = np.expand_dims(test_img,axis=0)
# result = classifier.predict(test_img)
# train_set.class_indices
# print(result[0][0])
# if result[0][0] == 1:
#         prediction = 'green'
#
# elif result[0][1] == 1:
#         prediction = 'red'
# else:
#         prediction = 'yellow'
#
#
# print(prediction)
#
#
#
#
# import cv2
#
# cap = cv2.VideoCapture(0)
#
# i = 0
# while(cap.isOpened()):
#
#         _,frame = cap.read()
#
#         #test_img = image.load_img(frame, target_size=(64, 64))
#         i  = i + 1
#
#         cv2.imshow("shashank",frame)
#         #classifier.predict(test_img)
#         if i % 50 == 0:
#                 cv2.imwrite("shashi" + str(i) + ".jpg",frame)
#                 cv2.imshow("shashi" + str(i) + ".jpg", frame)
#
#
#
#         #cv2.imshow("shashi" + str(i) +".jpg" , frame)
#
#         if cv2.waitKey(1) & 0xff == ord('q'):
#                 break
#
# cap.release()
# cv2.destroyAllWindows()
#
#
# img = cv2.imread('shashi2.jpg',0)
#
# cv2.imshow("shashank",img)
# k = cv2.waitKey(0)
# cv2.imwrite("loo",img)
# cv2.destroyAllWindows()
#
#
#
#
#
#
#
#
#
#
#
#
# import cv2
#
# print(cv2.__version__)
# classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
# image =  cv2.imread('shashi.png')
#
# boxes = classifier.detectMultiScale(image)
# print("loo shashi here im")
#
# for box in boxes:
#     x,y,width,height = box
#     x2,y2 = x + width , y + height
#     cv2.rectangle(image,(x,y),(x2,y2),(255,0,0),thickness=5)
#     print("hahah")
#     print(box)
# print("finally")
# cv2.imshow('shashank',image)
# k = cv2.waitKey(0)
# cv2.destroyAllWindows()
# print("came out")

#
# import cv2
#
# cap = cv2.VideoCapture(0)
#
# classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
#
# def detect(image,image2):
#     boxes = classifier.detectMultiScale(image,1.3,5)
#     for box in boxes:
#         x,y,he,wi = box
#         x2,y2 = x + he,y+wi
#         cv2.rectangle(image2,(x,y),(x2,y2),(255,0,0),2)
#         cv2.putText(image2,"shashank",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1,cv2.LINE_AA)
#     return image2
#
# while(cap.isOpened()):
#     _,frame = cap.read()
#     grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     image = detect(grey,frame)
#     cv2.imshow("shashank",image)
#
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
# import matplotlib.pyplot as plt
# import cv2
# image = plt.imread('shashi50.jpg')
# cv2.imshow("png",image)
# k = cv2.waitKey(0)
# cv2.destroyAllWindows()



'''Face Dectection using Deep Larning pretrained model
The network uses a cascade structure with three networks; first the image is rescaled to a range of different sizes (called an image pyramid), then the first model (Proposal Network or P-Net) proposes candidate facial regions, the second model (Refine Network or R-Net) filters the bounding boxes, and the third model (Output Network or O-Net) proposes facial landmarks.
mtcnn library uses pretrained model
Detect_face function
This returns a list of dict object, each providing a number of keys for the details of each face detected, including:

‘box‘: Providing the x, y of the bottom left of the bounding box, as well as the width and height of the box.
‘confidence‘: The probability confidence of the prediction.
‘keypoints‘: Providing a dict with dots for the ‘left_eye‘, ‘right_eye‘, ‘nose‘, ‘mouth_left‘, and ‘mouth_right‘.
'''

'''Still Image'''

# import cv2
# from matplotlib import pyplot as plt
# from mtcnn.mtcnn import MTCNN
#
# #image = plt.imread('shashi300.jpg')
# image = cv2.imread('shashi300.jpg')
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#
# detector = MTCNN()
#
# faces = detector.detect_faces(image)
#
# for face in faces:
#     x,y,wid,he = face['box']
#     cv2.rectangle(image,(x,y),(x+wid,y+he),(255,0,0),2,cv2.LINE_AA)
#
#     print(face)
#
# cv2.imshow("shashank",image)
# k = cv2.waitKey(0)
# cv2.destroyAllWindows()

'''Web CAam'''
#
#
import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

def detect(frame,image):
    faces = detector.detect_faces(image)
    for face in faces:
        x,y,wi,he = face['box']
        cv2.rectangle(frame,(x,y),(x+wi,y+he),(255,0,0),2,cv2.LINE_AA)
        cv2.putText(frame,"shashi",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv2.LINE_AA)

    return frame

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _,frame = cap.read()
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = detect(frame,image)
    cv2.imshow("shashank",frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




##print("hello shashank")



#import cv2,glob





# import cv2
#
# cap = cv2.VideoCapture(0)
#
# classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
# print(classifier)
# def detect(image,image2):
#     boxes = classifier.detectMultiScale(image,1.3,5)
#     for box in boxes:
#         x,y,he,wi = box
#         x2,y2 = x + he,y+wi
#         cv2.rectangle(image2,(x,y),(x2,y2),(255,0,0),2)
#         cv2.putText(image2,"shashank",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1,cv2.LINE_AA)
#     return image2
#
# while(cap.isOpened()):
#     _,frame = cap.read()
#     grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     image = detect(grey,frame)
#     cv2.imshow("shashank",image)
#
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#




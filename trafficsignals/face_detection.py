from flask import Flask
import cv2
app = Flask(__name__)

@app.route("/")
def hello():


    cap = cv2.VideoCapture(0)

    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    print(classifier)
    def detect(image,image2):
        boxes = classifier.detectMultiScale(image,1.3,5)
        for box in boxes:
            x,y,he,wi = box
            x2,y2 = x + he,y+wi
            cv2.rectangle(image2,(x,y),(x2,y2),(255,0,0),2)
            cv2.putText(image2,"shashank",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1,cv2.LINE_AA)
        return image2

    while(cap.isOpened()):
        _,frame = cap.read()
        grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        image = detect(grey,frame)
        cv2.imshow("shashank",image)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


    return "Hello, World!"


if __name__ == '__main__':

    app.run(debug=True)



#
#
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



'''Another Method......'''

# import cv2
# from mtcnn.mtcnn import MTCNN
# detector = MTCNN()
#
# def detect(frame,image):
#     faces = detector.detect_faces(image)
#     for face in faces:
#         x,y,wi,he = face['box']
#         cv2.rectangle(frame,(x,y),(x+wi,y+he),(255,0,0),2,cv2.LINE_AA)
#     return frame
#
# cap = cv2.VideoCapture(0)
#
# while cap.isOpened():
#     _,frame = cap.read()
#     image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     frame = detect(frame,image)
#     cv2.imshow("shashank",frame)
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

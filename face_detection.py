import cv2
import sys

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)

while True:

	# capture frame by frame
	retval, frame = video_capture.read()
	# convert to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect features specified in haarcascade_frontalface_default.xml
	faces = face_cascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(35,35)
	)

	# draw a rectangle around recognized faces
	for(x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (40,167,69), 2)

	# display the resulting frame
	cv2.imshow('Video', frame)

	# exit the camera
	if cv2.waitKey(1) & 0xFF == ord('q'):
		sys.exit()

#print('test')

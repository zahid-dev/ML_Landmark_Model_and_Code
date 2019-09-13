import cv2
import dlib
import os
import boto3
import botocore
import urllib
import numpy as np
import json
from botocore.client import Config
import s3transfer
from subprocess import call
import zipfile
from PIL import Image
import os.path


call('rm -rf /tmp/*', shell=True)
facedata = {}
renderviews = 0
predictor = dlib.shape_predictor("dlib_predictor1.dat")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
CAM = np.zeros((3, 3),dtype=np.float64);
CAM[0]=[653.0839199346667, 0.0, 319.5]
CAM[1]=[0, 653.0839199346667, 239.5]
CAM[2]=[0,0,1]
cam_matrix = np.ascontiguousarray(CAM[:,:3]).reshape((3,3));

DIST = np.zeros((5, 1),dtype=np.float64);
DIST[0]=[0.0708346336844071]
DIST[1]=[0.06914019373717535]
DIST[2]=[0]
DIST[3]=[0]
DIST[4]=[-1.307346032368929]
dist_coeffs = np.ascontiguousarray(DIST[:,:1]).reshape((5,1))


count1=0;
count2=0;
count3=0;
	

B=np.matrix('6.825897, 6.760612, 4.402142; 1.330353, 7.122144, 6.903745;-1.330353, 7.122144, 6.903745;-6.825897, 6.760612, 4.402142;5.311432, 5.485328, 3.987654;1.789930, 5.393625, 4.413414;-1.789930, 5.393625, 4.413414;-5.311432, 5.485328, 3.987654;2.005628, 1.409845, 6.165652;-2.005628, 1.409845, 6.165652;2.774015, -2.080775, 5.048531;-2.774015, -2.080775, 5.048531;0.000000, -3.116408, 6.097667;0.000000, -7.415691, 4.070434',dtype=np.float64)
object_points = np.ascontiguousarray(B[:,:3]).reshape((14,3))
rvec=np.zeros((3,1));
V =np.zeros((3,3),dtype=np.float64);
rotation_mat = np.ascontiguousarray(V[:,:3]).reshape((3,3))
tvec=np.zeros((3,1));
P=np.zeros((3,4),dtype=np.float64);
pose_mat = np.ascontiguousarray(P[:,:4]).reshape((3,4))
euler_angle=np.zeros((3,1));

A=np.matrix('10.0, 10.0, 10.0;10.0, 10.0, -10.0;10.0, -10.0, -10.0;10.0, -10.0, 10.0;-10.0, 10.0, 10.0;-10.0, 10.0, -10.0;-10.0, -10.0, -10.0;-10.0, -10.0, 10.0')
reprojectsrc = np.ascontiguousarray(A[:,:3]).reshape((8,3))

L=np.zeros((2,8),dtype="int");
reprojectdst = np.ascontiguousarray(L[:,:8].reshape(2,8),dtype=np.int)
out_intrinsics=np.zeros((3,3),dtype=np.float64);
out_rotation=np.zeros((3,3),dtype=np.float64);
out_translation=np.zeros((3,1),dtype=np.float64);



	
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords


# detect the face rectangle 
def detect(img, cascade = face_cascade , minimumFeatureSize=(20, 20)):
    if cascade.empty():
        raise (Exception("There was a problem loading your Haar Cascade xml file."))
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)
    
    # if it doesn't return rectangle return array
    # with zero lenght
    if len(rects) == 0:
        return []

    #  convert last coord from (width,height) to (maxX, maxY)
    rects[:, 2:] += rects[:, :2]

    return rects

def Performance(frame):
    global renderviews
    global facedata
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect the face at grayscale image
    te = detect(gray, minimumFeatureSize=(80, 80))

    # if the face detector doesn't detect face
    # return None, else if detects more than one faces
    # keep the bigger and if it is only one keep one dim
    if len(te) == 0:
        return None
    elif len(te) > 1:
        face = te[0]
    elif len(te) == 1:
        [face] = te

    # keep the face region from the whole frame
    face_rect = dlib.rectangle(left = int(face[0]), top = int(face[1]),
                                right = int(face[2]), bottom = int(face[3]))
    
    # determine the facial landmarks for the face region
    shape = predictor(gray, face_rect)
    shape = shape_to_np(shape)

    #  grab the indexes of the facial landmarks for the left and
    #  right eye, respectively
    
    """for (x, y) in shape:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)"""
    #D=np.array([[shape[17,0],shape[17,1]],[shape[21,0],shape[21,1]],[shape[22,0],shape[22,1]],[shape[26,0],shape[26,1]],[shape[36,0],shape[36,1]],[shape[39,0],shape[39,1]],[shape[42,0],shape[42,1]],[shape[45,0],shape[45,1]],[shape[31,0],shape[31,1]],[shape[35,0],shape[35,1]],[shape[48,0],shape[48,1]],[shape[54,0],shape[54,1]],[shape[57,0],shape[57,1]],[shape[8,0],shape[8,1]]],dtype=np.float64)
    D=np.array([[shape[9,0],shape[9,1]],[shape[14,0],shape[14,1]],[shape[15,0],shape[15,1]],[shape[19,0],shape[19,1]],[shape[30,0],shape[30,1]],[shape[33,0],shape[33,1]],[shape[37,0],shape[37,1]],[shape[40,0],shape[40,1]],[shape[25,0],shape[25,1]],[shape[29,0],shape[29,1]],[shape[43,0],shape[43,1]],[shape[50,0],shape[50,1]],[shape[53,0],shape[53,1]],[shape[66,0],shape[66,1]]],dtype=np.float64)
    image_pts = np.ascontiguousarray(D[:,:2]).reshape((14,2))

 #cv2.imshow("Output", frame)
 #cv2.waitKey(0)
 #image_pts=np.matrix('shape.part(17).x, shape.part(17).y;shape.part(21).x, shape.part(21).y;shape.part(22).x, shape.part(22).y;shape.part(26).x, shape.part(26).y;shape.part(36).x, shape.part(36).y;shape.part(39).x, shape.part(39).y;shape.part(42).x, shape.part(42).y;shape.part(45).x, shape.part(45).y;shape.part(31).x, shape.part(31).y;shape.part(35).x, shape.part(35).y;shape.part(48).x, shape.part(48).y;shape.part(54).x, shape.part(54).y;shape.part(57).x, shape.part(57).y;shape.part(8).x, shape.part(8).y')      

 
  #print(image_pts)
    (retval,rvec, tvec) = cv2.solvePnP(object_points, image_pts, cam_matrix, dist_coeffs)
  
  
    reprojectdst,jac=cv2.projectPoints(reprojectsrc,rvec,tvec,cam_matrix, dist_coeffs)
  #print(tuple(np.int(np.vectorize(reprojectdst[0].ravel()))));
 
    (x1,y1)=tuple(reprojectdst[0].ravel())
    x1=int(x1)
    y1=int(y1)
    (x2,y2)=tuple(reprojectdst[1].ravel())
    x2=int(x2)
    y2=int(y2)
    (x3,y3)=tuple(reprojectdst[2].ravel())
    x3=int(x3)
    y3=int(y3)
    (x4,y4)=tuple(reprojectdst[3].ravel())
    x4=int(x4)
    y4=int(y4)
    (x5,y5)=tuple(reprojectdst[4].ravel())
    x5=int(x5)
    y5=int(y5)
    (x6,y6)=tuple(reprojectdst[5].ravel())
    x6=int(x6)
    y6=int(y6)
    (x7,y7)=tuple(reprojectdst[6].ravel())
    x7=int(x7)
    y7=int(y7)
    (x8,y8)=tuple(reprojectdst[7].ravel())
    x8=int(x8)
    y8=int(y8)
 
    cv2.line(frame, (x1,y1), (x2,y2), (0, 0, 255));
    cv2.line(frame, (x2,y2), (x3,y3), (0, 0, 255));
    cv2.line(frame, (x3,y3), (x4,y4), (0, 0, 255));
    cv2.line(frame, (x4,y4), (x1,y1), (0, 0, 255));
    cv2.line(frame, (x5,y5), (x6,y6), (0, 0, 255));
    cv2.line(frame, (x6,y6), (x7,y7), (0, 0, 255));
    cv2.line(frame, (x7,y7), (x8,y8), (0, 0, 255));
    cv2.line(frame, (x8,y8), (x5,y5), (0, 0, 255));
    cv2.line(frame, (x1,y1), (x5,y5), (0, 0, 255));
    cv2.line(frame, (x2,y2), (x6,y6), (0, 0, 255));
    cv2.line(frame, (x3,y3), (x7,y7), (0, 0, 255));
    cv2.line(frame, (x4,y4), (x8,y8), (0, 0, 255));
  #print(rvec)
    rotation_mat,jocbian=cv2.Rodrigues(rvec);
  #print(rotation_mat);
  #print(tvec);
    pose_mat=np.concatenate((rotation_mat,tvec),axis=1)
 
  #print(pose_mat)
    rotMatrX=np.zeros((3,3))
    rotMatrY=np.zeros((3,3))
    rotMatrZ=np.zeros((3,3))
    out_intrinsics, out_rotation, out_translation, rotMatrX, rotMatrY, rotMatrZ, euler_angle=cv2.decomposeProjectionMatrix(pose_mat)
  #euler_angles = cv.DecomposeProjectionMatrix(projMatrix=rmat, cameraMatrix=camera_matrix, rotMatrix, transVect, rotMatrX=None, rotMatrY=None, rotMatrZ=None)

    
    str1="Pitch: "+str(euler_angle[0])
    str2="Yaw: "+str(euler_angle[1])
    str3="Roll: "+str(euler_angle[2])
    cv2.putText(frame, str1, (50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2);
    cv2.putText(frame, str2, (50,60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2);
    cv2.putText(frame, str3, (50,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2);
    image_pts=np.delete;
    horizontal_img = cv2.flip( frame, 1 )

       
        
        
        

    
   
      
    return

# make the image to have the same format as at training 



def lambda_handler(event, context):
	call('rm -r /tmp/*', shell=True)
	global renderviews
	global facedata
	global count1
	global count2
	global count3 
	facedata = {}
	renderviews = 0
	frame=0
	ret=False
	count1=0;
	count2=0;
	count3=0;
	print "OpenCV version=", cv2.__version__
	print "dlib version=", dlib.__version__
	print "json version=", json.__version__
	print "np version=", np.__version__
	s31 = boto3.client('s3', config=Config(signature_version='s3v4'))
	s3 = boto3.resource('s3', config=Config(signature_version='s3v4'))
	if event:
		""""file_obj = event["Records"][0]
		filename = str(file_obj['s3']['object']['key'])
		filename = urllib.unquote_plus(filename)
		print("Filename: ", filename)"""
		bucket = s3.Bucket('YOUR BUCKET NAME')
		#object = bucket.Object(filename)
		#fileObj = s3.get_object(Bucket = "upload-trigger-zahid", Key=filename)
		#file_content = fileObj["Body"].read()
		
		filename = event['key']
		print filename
		bucket.download_file(filename, '/tmp/'+filename)
		#object.download_fileobj(f)
		
		camera = cv2.VideoCapture('/tmp/'+filename)
		#camera = cv2.VideoCapture(file_content)

	if camera.isOpened():
		while True and renderviews < 3:
			
			ret, frame = camera.read()
			if ret==True:
      
        # detect eyes

					Performance(frame)
			else:
				break;
     
	
		del(frame)

    # do a little clean up
	del(camera)
	print "context=", context
	print "event=", event
	

	return {"Payload" : filename + ".zip"}
	
	

if __name__ == "__main__":
	lambda_handler(42, 42)

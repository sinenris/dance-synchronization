import numpy as np
import math
#This function will return a list of all slope values formed between certain (X,Y) points.
#The slope is calculated using the slope formula for two points A(x1,y1), B(x2,y2):m_AB = (y2-y1)/(x2-x1)

#input: pose_landmarks: the pose landmarks for a given person.
#       image_height the height of the input image.
#       image_width: the width of the input image.
#output: a list of all the slope values.
def computeMVals(pose_landmarks, image_height, image_width):
    m_values = []
    
    #values for shoulder,elbow.
    x_shoulder_left = pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER].x * image_width
    y_shoulder_left = pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER].y * image_height

    x_elbow_left = pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ELBOW].x * image_width
    y_elbow_left = pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ELBOW].y * image_height


    #shoulder, elbow
    x_shoulder_right = pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER].x * image_width
    y_shoulder_right = pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER].y * image_height

    x_elbow_right = pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_ELBOW].x * image_width
    y_elbow_right = pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_ELBOW].y * image_height


    #hip, knee, left.
    x_hip_left = pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP].x * image_width
    y_hip_left = pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP].y * image_height

    x_knee_left = pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_KNEE].x * image_width
    y_knee_left = pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_KNEE].y * image_height

    #hip, knee, right.
    x_hip_right = pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_HIP].x * image_width
    y_hip_right = pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_HIP].y * image_height

    x_knee_right = pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_KNEE].x * image_width
    y_knee_right = pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_KNEE].y * image_height



    m_SHOULDER_ELBOW_left = (y_elbow_left - y_shoulder_left) / (x_elbow_left - x_shoulder_left)
    m_SHOULDER_ELBOW_right = (y_elbow_right - y_shoulder_right) / (x_elbow_right - x_shoulder_right)

    m_HIP_KNEE_left = (y_hip_left - y_knee_left)/(x_hip_left - x_knee_left)
    m_HIP_KNEE_right = (y_hip_right - y_knee_right)/(x_hip_right - x_knee_right)
    m_values.append([m_SHOULDER_ELBOW_left, m_SHOULDER_ELBOW_right, m_HIP_KNEE_left, m_HIP_KNEE_right])

    return m_values


#This function will convert the value of a slope into its degree equivalent. 
#input:     slope_list -> a list of slopes
#output:    angle_list -> a list of all angles calculated with respect to the slope value.
def computeAngle(slope_list):
    angle_list = []
    for slope in slope_list:
        angle_list.append(math.degrees(math.atan(slope)))
    return angle_list


#This function will return the standard deviation of all the angles( value calculated in respect to the slope).
#input:     m_values -> a matrix of m_values. each row represents the m_values of the angles of a single person.
#example : k people, n slopes:m_val =[ [ p1_m1, p1_m2, ..., p1_mn],
#                                      [ p2_m1, p2_m2, ..., p2_mn],
#                                       .. 
#                                       ..
#                                      [p1_m1, p2_m2, .... pk_mn] ]
#                           
#                   
#                           ]
#output:    vec_of_sums -> standard deviation across column [std_dev1, std_dev2, ...., std_devk]
def stdDev(m_values):
    #we transform it into numpy arrays.
    m_val_np = np.array(m_values)
    #we compute the angle vals.
    angles_list = np.apply_along_axis(computeAngle, 0, m_val_np)
    #summing over column to use in the stddev formula.
    sum_per_columns_angles = angles_list.sum(axis=0)
    #number of values.
    num_of_elems = np.size(m_val_np,0)
    #median = sum_per_columns//num_of_elems
    # vector of m values throughout frames. [m val, m val]. we compute std deviation
    #sum (xi- median)^2/N

    sum = 0
    vec_of_sums = []
    for (idx,column) in enumerate(angles_list.T):
        #we get the column
        sum = 0
        median = sum_per_columns_angles[idx] // num_of_elems
        for i in column:
            sum += (i-median)*(i-median)

        vec_of_sums.append(math.sqrt(sum//num_of_elems))

    return vec_of_sums

#Compute the sync quotient for a given frame.
def synchronizationQuotient(vec_of_devs):
    #print("The vec of devs is:", vec_of_devs)
    pts = 0
    total_no_points = len(vec_of_devs)
    for elem in vec_of_devs:
        if(elem < 10):
            pts += 1
    #print(pts/total_no_points)
    return pts/total_no_points


import matplotlib.pyplot as plt
#Return if two squares overlap.
def compareDist(dim1, dim2):
    #x1,y1
    p1_x1, p1_y1 = dim1[0], dim1[1], 

    #x2y2
    p1_x2, p1_y2 = dim1[0] + dim1[2], dim1[1] + dim1[3]

    #x3y3
    p2_x1, p2_y1 = dim2[0], dim2[1]

    #x4y4
    p2_x2, p2_y2 = dim2[0] + dim2[2], dim2[1] + dim2[3]

    # plt.scatter(p1_x1, p1_y1)
    # plt.scatter(p1_x2, p1_y2)
    # plt.scatter(p2_x1, p2_y1, color = 'hotpink')
    # plt.scatter(p2_x2, p2_y2, color = 'hotpink')
    # plt.show()

    if (p1_x1>=p2_x2) or (p1_y1>p2_y2) or (p2_y1>p1_y2) or (p2_x1>=p1_x2):
        return False
    return True
        

    


from ctypes import sizeof
import cv2
import numpy as np
import mediapipe as mp
import os
# Yolo
cdir = os.getcwd()
#weights = cdir +"\logistics\YOLO\yolov3.weights"

weights="c:\\Users\\sinen\\Desktop\\Dance-Q\\YOLO\\yolov3.weights"
#yvfs = cdir + "\logistics\YOLO\yolov3.cfg"
yvfs = "C:\\Users\\sinen\\Desktop\\Dance-Q\\YOLO\\yolov3.cfg"
net = cv2.dnn.readNet(weights, yvfs)
classes = []
#with open(cdir + "\logistics\YOLO\coco.names", "r") as f:
with open("C:\\Users\\sinen\\Desktop\\Dance-Q\\YOLO\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#Mediapipe
mpPose = mp.solutions.pose
#pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

pTime = 0

# Loading web cam
#camera = cv2.VideoCapture(0)
# Loading picture
#img = cv2.imread("two_people.jpg")

def process_video(videoname):
        
    camera = cv2.VideoCapture(videoname)


    # global variables
    pose_estimator = []
    pose_estimator_dim = []
    conf_scores = []
    person_tracker = []
    img_array = []
    #output = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
    result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (432, 768))
    frame_counter_sync = 0
    frame_counter = 0
    #while True:
    while (camera.isOpened()):
        success, img = camera.read()
        if not success:
            break
        img = cv2.resize(img, None, fx=0.6, fy=0.6)
        #success, img = cv2.imread("two_people.jpg")
        height, width, channels = img.shape

        # Yolo to detect objects
        blob = cv2.dnn.blobFromImage(img, 1.0/255, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                if class_id == 0: #or class_id == 32:
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        conf_scores.append(float(confidence))
                        class_ids.append(class_id) 

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                #################################################################
                #########################################################
                if(len(pose_estimator)==0): 
                    pose = mpPose.Pose(min_detection_confidence=0.5,     
                                min_tracking_confidence=0.5)
                    pose_estimator.append(pose)
                    pose_estimator_dim.append([x,y,w,h])
                    person_tracker.append("p1")                           
                elif(len(pose_estimator)>=1):
                    overlaps = False
                    for idx, dim in enumerate(pose_estimator_dim):
                        if compareDist(dim,[x,y,w,h]):
                            overlaps = True
                            pose_estimator_dim[idx] = [x,y,w,h]
                    
                    if not overlaps:
                        pose = mpPose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
                        pose_estimator.append(pose)    
                        pose_estimator_dim.append([x,y,w,h])  
                #################################################################

        #here we append the m values
        all_m_vals = []
        for i in range(len(pose_estimator)):
            #if class_ids[i] == 0:
                #print(len(pose_estimator))
                x, y, w, h = pose_estimator_dim[i]
                crop_img = img[abs(y):abs(y)+h+20, abs(x):abs(x)+w+20]
                imgRGB = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                
                #results = pose.process(crop_img)
                results = pose_estimator[i].process(crop_img)
                if results.pose_landmarks:
                    #here i get the pose landmarks.
                    all_m_vals.extend(computeMVals(results.pose_landmarks, height,width))
                    mpDraw.draw_landmarks(crop_img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        h, w,c = crop_img.shape
                        #print(id, lm)
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        cv2.circle(crop_img, (cx, cy), 5, (255,0,0), cv2.FILLED)
                #Box and label
                #label = str(classes[class_ids[i]])
                label = 'person'
                color = colors[1]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 15), font, 1.2, color, 3)
                #output.write(img)

        try:

            frame_counter_sync += synchronizationQuotient(stdDev(all_m_vals))
            frame_counter += 1
        except:
            pass
        cv2.imshow("Image", img)
        result.write(img)
        #input("Enter")
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    #cv2.imwrite("result2.jpg", img)
    camera.release()
    result.release()
    cv2.destroyAllWindows()
    print("res ", frame_counter_sync/frame_counter)
    return frame_counter_sync/frame_counter


import cv2
import time
import numpy as np
import sys
from glob import iglob
import pickle

MODE = "MPI" # Keypoint Detection Dataset

if MODE is "COCO":
    protoFile = "pose_models/pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose_models/pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
                  [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

elif MODE is "MPI":
    # specifies the architecture of the neural network – how the different layers are arranged etc.
    protoFile = "pose_models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    # stores the weights of the trained model, trained on MPII dataset
    weightsFile = "pose_models/pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
                  [11, 12], [12, 13]]

inWidth = 368
inHeight = 368
threshold = 0.1

frame_height = 0
frame_width = 0

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

input_dir = sys.argv[1] + '*'
output_name = input_dir.split('/')[2] + input_dir.split('/')[3][:-8]

photo_ind = 0
skeleton_dict = {}
for f in iglob(input_dir):
    print(f)
    input_source = f
    cap = cv2.VideoCapture(input_source)
    hasFrame, frame = cap.read()

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    while cv2.waitKey(1) < 0 and hasFrame:
        t = time.time()

        frameCopy = np.copy(frame)
        blank_frame = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)

        if not hasFrame:
            cv2.waitKey()
            break

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        # converted to a input blob (like Caffe) so that it can be fed to the network
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        # makes a forward pass through the network, i.e. making a prediction
        output = net.forward() # 4D matrix, 1: image ID, 2: index of a keypoint, 3: height, 4: width of output map

        H = output.shape[2]
        W = output.shape[3]
        # Empty list to store the detected keypoints
        points = []

        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold:
                cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)

        # Add this skeletons points to our dictionary
        skeleton_dict[photo_ind] = []
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            
            if points[partA] and points[partB]:
                skeleton_dict[photo_ind].append((points[partA], points[partB]))

                #cv2.line(blank_frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                #cv2.circle(blank_frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                #cv2.circle(blank_frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8,
                (255, 50, 0), 2, lineType=cv2.LINE_AA)
        # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        # cv2.imshow('Output-Keypoints', frameCopy)
        cv2.imshow('Output-Skeleton', frame)

        #vid_writer.write(blank_frame)
        hasFrame, frame = cap.read()

    #vid_writer.release()
    photo_ind += 1

# Now, compile a single skeleton that is the average of all of the read in
# skeletons
averages = {i: np.zeros((2, 2)) for i in range(len(skeleton_dict[0]))}
for s in skeleton_dict:
    joints = skeleton_dict[s]

    # Each joints collection is a list of tuples
    for j in range(len(joints)):
        averages[j][0][0] += joints[j][0][0]
        averages[j][0][1] += joints[j][0][1]
        averages[j][1][0] += joints[j][1][0]
        averages[j][1][1] += joints[j][1][1]

for j in range(len(skeleton_dict[0])):
    for r in range(2):
        for c in range(2):
            averages[j][r][c] = int(averages[j][r][c] / len(skeleton_dict))


# Finally, write our average skeleton for this directory onto a blank frame
vid_writer = cv2.VideoWriter('output_jpgs/average_' + output_name + '.jpg',
                         cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
                         10, (1400, 1800))

blank_frame = np.zeros((1800, 1400, 3), np.uint8)
i = 0

center_point = tuple(averages[1][0])

# Normalize the center point to be half the width of the image,
# and one-third the height.
x_diff = (blank_frame.shape[1] // 2) - center_point[0]
y_diff = (blank_frame.shape[0] // 4) - center_point[1]

average_joints = {i: point for i in range(len(points))}
for key, value in averages.items():
    pointA = [int(val) for val in value[0]]
    pointB = [int(val) for val in value[1]]

    pointA[0] = int(pointA[0] + x_diff)
    pointA[1] = int(pointA[1] + y_diff)
    pointB[0] = int(pointB[0] + x_diff)
    pointB[1] = int(pointB[1] + y_diff)

    pointA = tuple(pointA)
    pointB = tuple(pointB)

    cv2.line(blank_frame, pointA, pointB, (0, 255, 255), 3, lineType=cv2.LINE_AA)
    if pointA == center_point:
        cv2.circle(blank_frame, pointA, 8, (255, 0, 255), thickness=-1, lineType=cv2.FILLED)
    else:
        cv2.circle(blank_frame, pointA, 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.circle(blank_frame, pointB, 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    
    # Add the points from this line to the joint dictionary, if they haven't
    # already been added
    first = POSE_PAIRS[key][0]
    second = POSE_PAIRS[key][1]
    average_joints[first] = pointA
    average_joints[second] = pointB
    i += 1


with open('average_joints/average_joints_' + output_name + '.pickle', 'wb') as handle:
    pickle.dump(average_joints, handle, protocol=pickle.HIGHEST_PROTOCOL)

vid_writer.write(blank_frame)
vid_writer.release()




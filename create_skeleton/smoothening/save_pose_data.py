import sys
import cv2, numpy as np, csv
import pandas as pd

outfile_path = './output_csvs/' + sys.argv[1].split('/')[-1][:-4] + '.csv'

# Load the structure and weights of the NN from OpenPose
protoFile = "../pose_models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "../pose_models/pose/mpi/pose_iter_160000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

data, input_width, input_height, threshold, frame_number = [], 368, 368, 0.1, 0

input_source = sys.argv[1]
cap = cv2.VideoCapture(input_source)

# use the previous location of the body part if the model is wrong
previous_x, previous_y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

while True:

    ret, img = cap.read()
    if not ret: break

    # get the image shape
    img_width, img_height = img.shape[1], img.shape[0]

    # get a blob from the image
    inputBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (input_width, input_height),(0, 0, 0), swapRB=False, crop=False)

    # set the input and perform a forward pass
    net.setInput(inputBlob)
    output = net.forward()

    # get the output shape
    output_width, output_height = output.shape[2], output.shape[3]

    # Empty list to store the detected keypoints
    x_data, y_data = [], []

    # Iterate through the body parts
    for i in range(15):

        # find probability that point is correct
        _, prob, _, point = cv2.minMaxLoc(output[0, i, :, :])        

        # Scale the point to fit on the original image
        x, y = (img_width * point[0]) / output_width, (img_height * point[1]) / output_height        

        # Is the point likely to be correct?
        if prob > threshold:
            x_data.append(x)
            y_data.append(y)
            xy = tuple(np.array([x,y], int))
            cv2.circle(img, xy, 5, (25,0,255), 5)
        # No? us the location in the previous frame
        else:
            x_data.append(previous_x[i])
            y_data.append(previous_y[i])

    # add these points to the list of data
    data.append(x_data + y_data)
    previous_x, previous_y = x_data, y_data
    frame_number+=1
    # use this break statement to check your data before processing the whole video
    #if frame_number == 300: break
    print(frame_number)
    
    cv2.imshow('img', img)
    k = cv2.waitKey(1)
    if k == 27: break

# write the data to a .csv file
df = pd.DataFrame(data)
df.to_csv(outfile_path, index = False)
print('save complete')


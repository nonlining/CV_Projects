#-------------------------------------------------------------------------------
# Name:        facial keypoint detection
# Purpose:
#
# Author:      nonlining
#
# Created:     27/07/2018
# Copyright:   (c) nonlining 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import cv2
import torch
import numpy as np
from models import Net
import torch.cuda


def connectingWebCam():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
    net = Net().cuda()
    net.load_state_dict(torch.load('saved_models/keypoints_model_1.pt'))
    net.eval()
    count = 0
    prev_h = 0
    prev_w = 0
    prev_x = 0
    prev_y = 0

    while(True):
        count += 1
        if(count % 2 == 0):
            continue

        ret, frame = cap.read()
        faces = face_cascade.detectMultiScale(frame, 1.2, 2)
        frame_with_detections = frame.copy()


        roi = frame_with_detections

        face_images = []

        if(len(faces) > 1):
            sorted(faces,key=lambda l:l[3], reverse=True)

        #for (x,y,w,h) in faces:
        if len(faces) >= 1:
            (x,y,w,h) = faces[0]

            if(prev_h != 0 and abs(prev_h - h)/float(prev_h) < 0.15):
                h = prev_h
                y = prev_y
            if(prev_w != 0 and abs(prev_w - w)/float(prev_w) < 0.15):
                w = prev_w
                x = prev_x

            if w < h:
                w = h

            cv2.rectangle(frame_with_detections,(x,y),(x+w,y+h),(255,0,0),3)

            roi = frame[y: y+h ,x:x+w]

            roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

            if type(roi).__module__ != np.__name__:
                continue

            roi = (roi / 255.).astype(np.float32)

            ori_h, ori_w = roi.shape

            roi = cv2.resize(roi, (224, 224))

            if len(roi.shape) == 2:
                roi = np.expand_dims(roi, axis=0)
            else:
                roi = np.rollaxis(roi, 2, 0)

            roi = np.expand_dims(roi, axis=0)
            roi = torch.from_numpy(roi).type(torch.FloatTensor).cuda()

            results = net.forward(roi)
            results = results.view(results.size()[0], 68, -1).cpu()
            predicted_key_pts = results[0].data
            predicted_key_pts = predicted_key_pts.cpu().numpy()
            predicted_key_pts = predicted_key_pts*50.0+100

            for p in predicted_key_pts:
                cv2.circle(frame_with_detections,(x + int(p[0]*ori_w/224.),y + int(p[1]*ori_h/224.)), 2, (0,0,255), -1)

            prev_h = h
            prev_w = w
            prev_x = x
            prev_y = y

        cv2.imshow('frame', frame_with_detections)

        if cv2.waitKey(1) & 0xFF == 27: # esc to quit
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    connectingWebCam()

if __name__ == '__main__':
    main()

import cv2

### read coco.names file for the class names ###
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
f.close()
# print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weigthPath = 'frozen_inference_graph.pb'
### create the object detection model ###
net = cv2.dnn_DetectionModel(weigthPath, configPath)
# model settings
net.setInputSize(320,320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def objDetect(img, threshold, nms_threshold, objects=[], display=True):
    # send image to the model
    classIds, confs, bbox = net.detect(img, confThreshold=threshold, nmsThreshold=nms_threshold)
    objInfo = []
    if len(classIds) !=0:
        if len(objects) == 0: objects = classNames
        # Draw object detection results in the image
        for classIds, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classIds-1]
            if className in objects:
                objInfo.append([box, className])
                if display:
                    cv2.rectangle(img, box, (0,255,0), 2)
                    textLen = len(className)
                    # cv2.rectangle(img, (box[0], box[1]), (box[0]+textLen*15+15, box[1]+30),(127,127,127), -1)
                    cv2.putText(img, className.upper(), (box[0]+10,box[1]+20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,255,0),2)
                    conf = str(round(confidence*100, 2)) + ' %'
                    cv2.putText(img, conf, (box[0]+textLen*15+15,box[1]+20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,255,0),2)
    return img, objInfo

if __name__ == "__main__":
    threshold = 0.6  # threshold to detect object
    nms_threshold = 0.5
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # capture image width
    cap.set(4, 480)  # capture image height
    while True:
        success, img = cap.read()
        result, objInfo = objDetect(img, threshold, nms_threshold)
        cv2.imshow('Image', result)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

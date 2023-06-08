import cv2
import os
import time
# import serial

# Load pre-trained PPE YOLOv3 model and class names
net = cv2.dnn.readNet("yolov3-tiny_final.weights", "yolov3-tiny.cfg")
classes = [4]
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Path dari Data
cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
# Filtering with Cascade
faceCascade = cv2.CascadeClassifier(cascPathface)

# Akses Webcam
cap = cv2.VideoCapture(0)

# Variables FPS
start_time = time.time()
frame_count = 0
frame_size = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        faceROI = frame[y:y+h, x:x+w]

    # PPE detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # Process the detections
    has_ppe = False
    for detection in outs:
        for detection_info in detection:
            scores = detection_info[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in ["helm", "masker", "baju_pelindung","sepatu safety"]:
                has_ppe = True
                break

    # Display the frame
    if has_ppe:
        cv2.putText(frame, "APD Terdeteksi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "APD Tidak Terdeteksi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Calculate FPS
    frame_count += 1
    if frame_count >= 10:  # Update FPS every 10 frames
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Video', frame)

    # Check akses ruangan
    if not has_ppe:
        print("Akses Masuk Ruangan Ditolak!")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the windows
cap.release()
cv2.destroyAllWindows()


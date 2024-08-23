import cv2
import numpy as np
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

ret, frame1 = cap.read()
ret, frame2 = cap.read()

status_displayed = False  

while True:
    ret, frame = cap.read()
    if not ret:
        break  
    centroids = []
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    movement_detected = False
    cv2.putText(frame, "Dot: Average Movement", (370, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 2000:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 4)
                
                
            movement_detected = True
            
    
    
    if not movement_detected:
        cv2.rectangle(frame1, (10, 0), (400, 40), (0, 0, 0), -1)
        cv2.putText(frame, "Status: No Movement", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        if centroids:
            avg_centroid = np.mean(centroids, axis=0).astype(int)
            
            cv2.circle(frame, (avg_centroid[0], avg_centroid[1]), 10, (0, 0, 255), -1)
        cv2.rectangle(frame1, (10, 0), (400, 40), (0, 0, 0), -1)
        cv2.putText(frame, "Status: Movement", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    resized_frame = cv2.resize(frame, (2560, 1620), interpolation=cv2.INTER_AREA)
    
    cv2.imshow('feed', resized_frame)
    
    frame1 = frame2
    ret, frame2 = cap.read()
    
    if cv2.waitKey(1) == 27:  # Escape key to break
        break

cap.release()
cv2.destroyAllWindows()

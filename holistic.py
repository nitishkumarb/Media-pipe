#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system('pip install mediapipe opencv-python')


# In[1]:


import cv2
import mediapipe as mp
import pandas as pd
import os 
import uuid


# In[2]:


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


# In[ ]:


import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Initialize Drawing Utilities
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Initialize Video Capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    success, image = cap.read()
    if not success:
        break

    # Convert image to RGB and process with MediaPipe Face Mesh
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw mesh on image
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    # Show image
    cv2.imshow("Face Mesh", image)

    # Wait for key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release Video Capture and destroy windows
cap.release()
cv2.destroyAllWindows()
# Wait for key press
key = cv2.waitKey(1)
if key == ord('q'):
    # Release Video Capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()
    break


# In[ ]:


import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize Drawing Utilities
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Initialize Video Capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    success, image = cap.read()
    if not success:
        break

    # Convert image to RGB and process with MediaPipe Hands
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw landmarks on image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    # Show image
    cv2.imshow("Hand Detection", image)

    # Wait for key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        # Release Video Capture and destroy windows
        cap.release()
        cv2.destroyAllWindows()
        break

# Release Video Capture and destroy windows
cap.release()
cv2.destroyAllWindows()


# In[3]:


import cv2
import mediapipe as mp

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Initialize Drawing Utilities
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Initialize Video Capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    success, image = cap.read()
    if not success:
        break

    # Convert image to RGB and process with MediaPipe Holistic
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw face landmarks on image
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

    # Draw pose landmarks on image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.pose_landmarks,
            connections=mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

    # Draw hand landmarks on image
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

    # Show image
    cv2.imshow("Holistic Detection", image)

    # Wait for key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        # Release Video Capture and destroy windows
        cap.release()
        cv2.destroyAllWindows()
        break

# Release Video Capture and destroy windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:





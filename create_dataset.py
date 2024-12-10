import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=.3)

DATA_DIR = './data'


data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        path = os.path.join(DATA_DIR, dir_, img_path)
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            
            temp_data = []

            for hand_landmarks in results.multi_hand_landmarks:
                # mp_drawing.draw_landmarks(
                #     img_rgb,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing.DrawingSpec(color=(0,0,255), thickness=5, circle_radius=5),
                #     mp_drawing.DrawingSpec(color=(0,255,0), thickness=10, circle_radius=10)
                # )

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    temp_data.append(x)
                    temp_data.append(y)

            data.append(temp_data)
            labels.append(dir_)

#         plt.figure()
#         plt.imshow(img_rgb)

# plt.show()
f = open('data.pickle', 'wb') # saving the data into a pickle file as a byte

pickle.dump({'data':data, 'labels': labels}, f)

f.close()

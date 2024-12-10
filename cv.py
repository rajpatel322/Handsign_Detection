import cv2
import mediapipe as mp
import pickle
import numpy as np

model_dict = pickle.load(open('./svm_model.pickle', 'rb'))
model = model_dict['model']


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=.3, max_num_hands=2)

labels_dict = {0:'1', 1:'2', 2:'3', 3:'4', 4:'5', 5:'6', 6:'7', 7:'8', 8:'9', 9:'10'}


temp_data = []
x_ = []
y_ = []


while True:

    temp_data.clear()
    x_.clear()
    y_.clear()


    ret, frame = cap.read()

    H,W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            
            
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=5, circle_radius=5),
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=10, circle_radius=10)
                )

                for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        temp_data.append(x)
                        temp_data.append(y)

                        x_.append(x)
                        y_.append(y)

            x1 = int(min(x_) * W)
            x2 = int(max(x_) * W)


            y1 = int(min(y_) * H)
            y2 = int(max(y_) * H)

            prediction = model.predict([np.asarray(temp_data)])

            predicted = labels_dict[int(prediction[0])]
            # print(predicted)
    
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), 5)
            cv2.putText(frame, predicted, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 8, (255,255,255), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if(cv2.waitKey(25) == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()

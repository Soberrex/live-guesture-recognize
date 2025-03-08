import cv2
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from scipy import stats
import mediapipe as mp
import shutil

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Constants
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'how'])
no_sequences = 30
sequence_length = 30
start_folder = 0

# Create directories and clean previous data
def setup_directories():
    # Remove existing data
    shutil.rmtree(DATA_PATH, ignore_errors=True)
    shutil.rmtree('Logs', ignore_errors=True)
    
    # Create fresh directories
    os.makedirs(DATA_PATH, exist_ok=True)
    for action in actions:
        action_dir = os.path.join(DATA_PATH, action)
        os.makedirs(action_dir, exist_ok=True)
        for seq in range(no_sequences):
            seq_dir = os.path.join(action_dir, str(seq))
            os.makedirs(seq_dir, exist_ok=True)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    # Draw face connections
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    
    # Draw pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    
    # Draw hands
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                   for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] 
                   for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] 
                 for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] 
                 for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def collect_keypoints():
    setup_directories()
    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False
    ) as holistic:
        
        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):
                    success = False
                    while not success:
                        ret, frame = cap.read()
                        if not ret:
                            continue
                            
                        image, results = mediapipe_detection(frame, holistic)
                        keypoints = extract_keypoints(results)
                        
                        if keypoints is not None and len(keypoints) == 1662:
                            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                            np.save(npy_path, keypoints)
                            success = True
                            
                            if frame_num == 0:
                                cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                                cv2.putText(image, f'Collecting {action} Video {sequence}', (15,12), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            else:
                                cv2.putText(image, f'Collecting {action} Video {sequence}', (15,12), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            
                            cv2.imshow('OpenCV Feed', image)
                            if cv2.waitKey(10) & 0xFF == ord('q'):
                                break
                        else:
                            print(f"Retrying {action} sequence {sequence} frame {frame_num}")
                            
        cap.release()
        cv2.destroyAllWindows()

def train_model():
    label_map = {label:num for num, label in enumerate(actions)}
    sequences, labels = [], []
    
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(actions.shape[0], activation='softmax')
    ])
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=2000, callbacks=[TensorBoard(log_dir='Logs')])
    model.save('action.h5')
    return model

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

def real_time_detection(model):
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    colors = [(245,117,16), (117,245,16), (16,117,245)]

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                
                if len(predictions) >= 10 and np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]
                
                image = prob_viz(res, actions, image, colors)
            
            cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Full workflow
    #setup_directories()
    
    # Step 1: Collect data
    #collect_keypoints()
    
    # Step 2: Train model
    #train_model()
    
    # Step 3: Run detection
    model = load_model('action.h5')
    real_time_detection(model)

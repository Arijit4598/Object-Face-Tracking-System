import cv2
import numpy as np
import threading
import smtplib
import speech_recognition as sr
import pyttsx3

camera_url = 'your camera url'
current_mode = None

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        print("Recognizing...")
        command = recognizer.recognize_google(audio)
        print(f"User: {command}")
        return command.lower()
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

def stop_code():
    global current_mode
    if current_mode:
        speak(f"Stopping {current_mode} module.")
        current_mode = None

def face_tracking():
    global current_mode
    current_mode = "face tracking"

    # Enable camera
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        current_mode = None
        return

    # import cascade file for facial recognition 
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while current_mode == "face tracking":
        success, img = cap.read()
        if not success:
            print("Error: Could not read frame from camera.")
            break

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Getting corners around the face
        faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbor
        # drawing bounding box around face
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow('face_detect', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_code()

    cap.release()
    cv2.destroyAllWindows()

def object_tracking():
    global current_mode
    current_mode = "object tracking"

    thres = 0.45  # Threshold to detect object
    nms_threshold = 0.5  # NMS

    # OpenCV DNN module setup
    configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weightsPath = "frozen_inference_graph.pb"

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # Load class names
    classFile = "coco.names"
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    # Open camera
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        current_mode = None
        return

    # Set camera properties
    cap.set(3, 1280)
    cap.set(4, 720)
    cap.set(10, 150)

    while current_mode == "object tracking":
        # Start Webcam
        success, image = cap.read()
        if not success:
            print("Error: Failed to capture frame.")
            break

        # Perform object detection
        classIds, confs, bbox = net.detect(image, confThreshold=thres)

        # Draw bounding boxes and labels
        if len(classIds) > 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                x, y, w, h = box
                cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                cv2.putText(image, classNames[classId - 1], (x + 10, y + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # Show output
        cv2.imshow("Output", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_code()

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def main():
    global current_mode
    speak("Hello, tell me which program you want to run. Option A Face tracking option B Object tracking or option C Fire tracking.")

    while True:
        command = listen()

        if "face tracking" in command:
            stop_code()
            threading.Thread(target=face_tracking).start()
        elif "object tracking" in command:
            stop_code()
            threading.Thread(target=object_tracking).start()
        elif "stop" in command:
            stop_code()
        elif "exit" in command:
            speak("Goodbye!")
            break
        else:
            speak("I'm sorry, I don't understand that command.")

if __name__ == "__main__":
    main()
#  Drone Surveillance Software

Welcome to **Drone Surveillance Software*, a Python-based real-time vision assistant that uses **OpenCV**, **speech recognition**, and **deep learning** to track **faces** and **objects** through live camera feeds — all **controlled by your voice**.



---

##  Features

-  **Voice-Controlled Interface** using Google Speech Recognition & pyttsx3  
-  **Face Tracking** with Haar Cascade  
-  **Object Detection** using SSD MobileNet (COCO model)  
-  **Live Camera Streaming**  
-  **Real-time stopping & switching** between modules  
-  **Threaded execution** to keep UI responsive  

---

## Tech Stack

- Python   
- OpenCV (cv2)  
- pyttsx3 (Text-to-Speech)  
- SpeechRecognition + Microphone  
- Pre-trained MobileNet SSD (COCO model)  
- Haarcascade Classifier  

---

## Lessons Learned

- Multithreading in Python for responsive interaction  
- Real-time video feed handling  
- Using pre-trained models in OpenCV DNN module  
- Integrating voice input/output with AI systems  
- Clean module switching with global state  

---

##  Example Use Case

```
Voice: "Object tracking"
>> Live camera opens with bounding boxes and labels for detected objects.

Voice: "Stop"
>> Tracking stops gracefully.

Voice: "Face tracking"
>> Camera starts detecting and highlighting faces.
```








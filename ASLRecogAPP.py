# import cv2
# import mediapipe as mp
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import tkinter as tk
# from tkinter import scrolledtext
# from PIL import Image, ImageTk

# # ---------------------------
# # Your existing processing code
# # ---------------------------

# def detect_skin(frame):
#     # Convert to YCrCb and equalize the luminance channel
#     ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
#     y_channel = ycrcb[:, :, 0]
#     y_eq = cv2.equalizeHist(y_channel)
#     ycrcb[:, :, 0] = y_eq

#     # Adjusted thresholds might be needed after equalization.
#     lower_skin = np.array([0, 133, 77], dtype=np.uint8)
#     upper_skin = np.array([255, 173, 127], dtype=np.uint8)
#     mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
#     # Noise reduction using morphological operations
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
#     # Optionally, keep only the largest contour (if needed)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contour_mask = np.zeros_like(mask)
#     if contours:
#         cv2.drawContours(contour_mask, contours, -1, 255, thickness=cv2.FILLED)
#     mask = cv2.bitwise_and(mask, contour_mask)
    
#     return mask

# # ---------------------------
# # Initialization
# # ---------------------------
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# classifier = Classifier("C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary2Model/VGG16_model.h5")

# offset = 45
# imgSize = 250
# labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

# mp_hands = mp.solutions.hands  # for landmark connections
# hand_connections = mp_hands.HAND_CONNECTIONS

# # To store the history of predictions
# prediction_history = []

# # ---------------------------
# # Tkinter UI Setup
# # ---------------------------
# root = tk.Tk()
# root.title("Sign Language Recognition")
# root.geometry("1200x800")  # Adjust window size as needed

# # Divide the window into two panels:
# # Left panel (biggest) for the main camera feed.
# left_frame = tk.Frame(root, width=800, height=800, bg="black")
# left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# # Right panel for the additional images and prediction history.
# right_frame = tk.Frame(root, width=400, height=800)
# right_frame.pack(side=tk.RIGHT, fill=tk.Y)

# # Label for the main camera feed (largest display)
# main_image_label = tk.Label(left_frame)
# main_image_label.pack(fill=tk.BOTH, expand=True)

# # Labels for the processed binary image and the hand landmarks.
# binary_label = tk.Label(right_frame)
# binary_label.pack(pady=5)

# landmarks_label = tk.Label(right_frame)
# landmarks_label.pack(pady=5)

# # Scrolled text widget to show prediction history.
# history_text = scrolledtext.ScrolledText(right_frame, width=40, height=20)
# history_text.pack(pady=5)
# history_text.configure(state='disabled')

# # ---------------------------
# # Update function for video frames and UI elements
# # ---------------------------
# def update_frame():
#     global prediction_history
#     success, img = cap.read()
#     if not success:
#         root.after(10, update_frame)
#         return

#     imgOutput = img.copy()
#     hands, img = detector.findHands(img, draw=False)
    
#     # Variables to hold images for right-panel display.
#     imgWhite_for_display = None
#     imgCrop_landmarked_for_display = None
    
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
#         y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
#         x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
#         imgCrop = img[y1:y2, x1:x2]
        
#         if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
#             # Draw hand landmarks on the cropped image (for visualization)
#             imgCrop_landmarked = imgCrop.copy()
#             if 'lmList' in hand:
#                 lm_list = hand['lmList']
#                 for lm in lm_list:
#                     cv2.circle(imgCrop_landmarked, (lm[0] - x1, lm[1] - y1), 4, (0, 0, 255), -1)
#                 for connection in mp_hands.HAND_CONNECTIONS:
#                     if connection[0] < len(lm_list) and connection[1] < len(lm_list):
#                         pt1 = (lm_list[connection[0]][0] - x1, lm_list[connection[0]][1] - y1)
#                         pt2 = (lm_list[connection[1]][0] - x1, lm_list[connection[1]][1] - y1)
#                         cv2.line(imgCrop_landmarked, pt1, pt2, (0, 0, 255), 2)
            
#             # Create the binary image from the cropped region.
#             binaryMask = detect_skin(imgCrop)
#             binary_result = np.zeros_like(imgCrop)
#             binary_result[binaryMask > 0] = [255, 255, 255]
            
#             # Overlay landmarks on the binary image.
#             if 'lmList' in hand:
#                 for lm in lm_list:
#                     cv2.circle(binary_result, (lm[0] - x1, lm[1] - y1), 4, (0, 0, 0), -1)
#                 for connection in mp_hands.HAND_CONNECTIONS:
#                     pt1 = (lm_list[connection[0]][0] - x1, lm_list[connection[0]][1] - y1)
#                     pt2 = (lm_list[connection[1]][0] - x1, lm_list[connection[1]][1] - y1)
#                     cv2.line(binary_result, pt1, pt2, (0, 0, 0), 2)
            
#             # Resize the binary image to a fixed size (while preserving aspect ratio)
#             aspectRatio = h / w
#             imgWhite = np.ones((imgSize, imgSize), np.uint8) * 0
#             if aspectRatio > 1:
#                 k = imgSize / h
#                 wCal = math.ceil(k * w)
#                 imgResize = cv2.resize(binary_result, (wCal, imgSize))
#                 wGap = math.ceil((imgSize - wCal) / 2)
#                 imgWhite[:, wGap:wCal + wGap] = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
#             else:
#                 k = imgSize / w
#                 hCal = math.ceil(k * h)
#                 imgResize = cv2.resize(binary_result, (imgSize, hCal))
#                 hGap = math.ceil((imgSize - hCal) / 2)
#                 imgWhite[hGap:hCal + hGap, :] = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
            
#             # Prepare for classification.
#             imgWhiteRGB = cv2.cvtColor(imgWhite, cv2.COLOR_GRAY2BGR)
#             prediction, index = classifier.getPrediction(imgWhiteRGB, draw=False)
            
#             # If the confidence is high enough, annotate the main image and record the prediction.
#             if prediction[index] > 0.75 and 0 <= index < len(labels):
#                 cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
#                               (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
#                 cv2.putText(imgOutput, labels[index], (x, y - 26),
#                             cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
#                 cv2.rectangle(imgOutput, (x - offset, y - offset),
#                               (x + w + offset, y + h + offset), (255, 0, 255), 4)
                
#                 # Append the prediction with its probability to the history.
#                 pred_text = f"{labels[index]}: {prediction[index]:.2f}"
#                 prediction_history.append(pred_text)
#                 if len(prediction_history) > 50:
#                     prediction_history = prediction_history[-50:]
#                 history_text.configure(state='normal')
#                 history_text.insert(tk.END, pred_text + "\n")
#                 history_text.see(tk.END)
#                 history_text.configure(state='disabled')
            
#             # Save images to display on the right panel.
#             imgWhite_for_display = imgWhite.copy()
#             imgCrop_landmarked_for_display = imgCrop_landmarked.copy()
    
#     # ---------------------------
#     # Convert OpenCV images to a format Tkinter can display.
#     # Main image (imgOutput) is in BGR, so convert to RGB.
#     imgOutput_rgb = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
#     img_pil = Image.fromarray(imgOutput_rgb)
#     img_tk = ImageTk.PhotoImage(image=img_pil)
#     main_image_label.imgtk = img_tk
#     main_image_label.configure(image=img_tk)
    
#     # Update the processed binary image display.
#     if imgWhite_for_display is not None:
#         imgWhite_rgb = cv2.cvtColor(imgWhite_for_display, cv2.COLOR_GRAY2RGB)
#         imgWhite_pil = Image.fromarray(imgWhite_rgb)
#         imgWhite_tk = ImageTk.PhotoImage(image=imgWhite_pil)
#         binary_label.imgtk = imgWhite_tk
#         binary_label.configure(image=imgWhite_tk)
#     else:
#         binary_label.configure(image='')
    
#     # Update the hand landmarks display.
#     if imgCrop_landmarked_for_display is not None:
#         imgCrop_rgb = cv2.cvtColor(imgCrop_landmarked_for_display, cv2.COLOR_BGR2RGB)
#         imgCrop_pil = Image.fromarray(imgCrop_rgb)
#         imgCrop_tk = ImageTk.PhotoImage(image=imgCrop_pil)
#         landmarks_label.imgtk = imgCrop_tk
#         landmarks_label.configure(image=imgCrop_tk)
#     else:
#         landmarks_label.configure(image='')
    
#     root.after(10, update_frame)

# # ---------------------------
# # Handle window closing to release the camera
# # ---------------------------
# def on_closing():
#     cap.release()
#     root.destroy()

# root.protocol("WM_DELETE_WINDOW", on_closing)

# # Start the update loop.
# update_frame()
# root.mainloop()


import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk

# ---------------------------
# Your existing processing code
# ---------------------------
def detect_skin(frame):
    # Convert to YCrCb and equalize the luminance channel
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0]
    y_eq = cv2.equalizeHist(y_channel)
    ycrcb[:, :, 0] = y_eq

    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Noise reduction using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Optionally, keep only the largest contour (if needed)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(mask)
    if contours:
        cv2.drawContours(contour_mask, contours, -1, 255, thickness=cv2.FILLED)
    mask = cv2.bitwise_and(mask, contour_mask)
    
    return mask

# ---------------------------
# Initialization
# ---------------------------
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary2Model/VGG16_model.h5")

offset = 45
imgSize = 250
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

mp_hands = mp.solutions.hands  # for landmark connections
hand_connections = mp_hands.HAND_CONNECTIONS

# To store the history of predictions
prediction_history = []

# ---------------------------
# Global variables for word typing
# ---------------------------
current_word = ""         # the word being built from the recognized letters
last_added_letter = ""    # the last letter that was added (to prevent repeats)
prev_hand_detected = False  # flag to detect transition from hand present to not

# ---------------------------
# Tkinter UI Setup
# ---------------------------
root = tk.Tk()
root.title("Sign Language Recognition")
root.geometry("1200x800")  # Adjust window size as needed

# Divide the window into two panels:
# Left panel (biggest) for the main camera feed.
left_frame = tk.Frame(root, width=800, height=800, bg="black")
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Right panel for the additional images and prediction history.
right_frame = tk.Frame(root, width=400, height=800)
right_frame.pack(side=tk.RIGHT, fill=tk.Y)

# Label for the main camera feed (largest display)
main_image_label = tk.Label(left_frame)
main_image_label.pack(fill=tk.BOTH, expand=True)

# Labels for the processed binary image and the hand landmarks.
binary_label = tk.Label(right_frame)
binary_label.pack(pady=5)

landmarks_label = tk.Label(right_frame)
landmarks_label.pack(pady=5)

# Scrolled text widget to show prediction history.
history_text = scrolledtext.ScrolledText(right_frame, width=40, height=20)
history_text.pack(pady=5)
history_text.configure(state='disabled')

# New label for showing the current word being typed.
word_var = tk.StringVar()
word_var.set("")
word_label = tk.Label(right_frame, textvariable=word_var, font=("Helvetica", 20))
word_label.pack(pady=10)

# ---------------------------
# Update function for video frames and UI elements
# ---------------------------
def update_frame():
    global current_word, last_added_letter, prev_hand_detected, prediction_history
    success, img = cap.read()
    if not success:
        root.after(10, update_frame)
        return

    imgOutput = img.copy()
    hands, img = detector.findHands(img, draw=False)
    
    # Variables to hold images for right-panel display.
    imgWhite_for_display = None
    imgCrop_landmarked_for_display = None

    # Flag to track if a hand is detected in the current frame.
    hand_detected = False
    if hands:
        hand_detected = True
        hand = hands[0]
        x, y, w, h = hand['bbox']
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]
        
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            # Draw hand landmarks on the cropped image (for visualization)
            imgCrop_landmarked = imgCrop.copy()
            if 'lmList' in hand:
                lm_list = hand['lmList']
                for lm in lm_list:
                    cv2.circle(imgCrop_landmarked, (lm[0] - x1, lm[1] - y1), 4, (0, 0, 255), -1)
                for connection in mp_hands.HAND_CONNECTIONS:
                    if connection[0] < len(lm_list) and connection[1] < len(lm_list):
                        pt1 = (lm_list[connection[0]][0] - x1, lm_list[connection[0]][1] - y1)
                        pt2 = (lm_list[connection[1]][0] - x1, lm_list[connection[1]][1] - y1)
                        cv2.line(imgCrop_landmarked, pt1, pt2, (0, 0, 255), 2)
            
            # Create the binary image from the cropped region.
            binaryMask = detect_skin(imgCrop)
            binary_result = np.zeros_like(imgCrop)
            binary_result[binaryMask > 0] = [255, 255, 255]
            
            # Overlay landmarks on the binary image.
            if 'lmList' in hand:
                for lm in lm_list:
                    cv2.circle(binary_result, (lm[0] - x1, lm[1] - y1), 4, (0, 0, 0), -1)
                for connection in mp_hands.HAND_CONNECTIONS:
                    pt1 = (lm_list[connection[0]][0] - x1, lm_list[connection[0]][1] - y1)
                    pt2 = (lm_list[connection[1]][0] - x1, lm_list[connection[1]][1] - y1)
                    cv2.line(binary_result, pt1, pt2, (0, 0, 0), 2)
            
            # Resize the binary image to a fixed size (while preserving aspect ratio)
            aspectRatio = h / w
            imgWhite = np.ones((imgSize, imgSize), np.uint8) * 0
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(binary_result, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(binary_result, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
            
            # Prepare for classification.
            imgWhiteRGB = cv2.cvtColor(imgWhite, cv2.COLOR_GRAY2BGR)
            prediction, index = classifier.getPrediction(imgWhiteRGB, draw=False)
            
            # If the confidence is high enough, annotate the main image and record the prediction.
            if prediction[index] > 0.75 and 0 <= index < len(labels):
                letter = labels[index]
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, letter, (x, y - 26),
                            cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)
                
                # Append the prediction with its probability to the history.
                pred_text = f"{letter}: {prediction[index]:.2f}"
                prediction_history.append(pred_text)
                if len(prediction_history) > 50:
                    prediction_history = prediction_history[-50:]
                history_text.configure(state='normal')
                history_text.insert(tk.END, pred_text + "\n")
                history_text.see(tk.END)
                history_text.configure(state='disabled')
                
                # ---------------------------
                # Update the current word being typed.
                # Only add the letter if it is different from the last added letter.
                if letter != last_added_letter:
                    current_word += letter
                    last_added_letter = letter
                    word_var.set(current_word)
                
            # Save images to display on the right panel.
            imgWhite_for_display = imgWhite.copy()
            imgCrop_landmarked_for_display = imgCrop_landmarked.copy()
    
    # ---------------------------
    # When no hand is detected, add a space once (only on the transition).
    # ---------------------------
    if not hand_detected:
        if prev_hand_detected:
            # Append a space if the last added character is not already a space.
            if last_added_letter != " ":
                current_word += " "
                last_added_letter = " "
                word_var.set(current_word)
        prev_hand_detected = False
    else:
        prev_hand_detected = True
    
    # ---------------------------
    # Convert OpenCV images to a format Tkinter can display.
    # Main image (imgOutput) is in BGR, so convert to RGB.
    imgOutput_rgb = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(imgOutput_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    main_image_label.imgtk = img_tk
    main_image_label.configure(image=img_tk)
    
    # Update the processed binary image display.
    if imgWhite_for_display is not None:
        imgWhite_rgb = cv2.cvtColor(imgWhite_for_display, cv2.COLOR_GRAY2RGB)
        imgWhite_pil = Image.fromarray(imgWhite_rgb)
        imgWhite_tk = ImageTk.PhotoImage(image=imgWhite_pil)
        binary_label.imgtk = imgWhite_tk
        binary_label.configure(image=imgWhite_tk)
    else:
        binary_label.configure(image='')
    
    # Update the hand landmarks display.
    if imgCrop_landmarked_for_display is not None:
        imgCrop_rgb = cv2.cvtColor(imgCrop_landmarked_for_display, cv2.COLOR_BGR2RGB)
        imgCrop_pil = Image.fromarray(imgCrop_rgb)
        imgCrop_tk = ImageTk.PhotoImage(image=imgCrop_pil)
        landmarks_label.imgtk = imgCrop_tk
        landmarks_label.configure(image=imgCrop_tk)
    else:
        landmarks_label.configure(image='')
    
    root.after(10, update_frame)

# ---------------------------
# Handle window closing to release the camera
# ---------------------------
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the update loop.
update_frame()
root.mainloop()

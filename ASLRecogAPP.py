import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import time  # for timing delays
import pyttsx3  # for text-to-speech

# Determine the resampling method for resizing images
try:
    resample_method = Image.Resampling.LANCZOS
except AttributeError:
    resample_method = Image.ANTIALIAS

# ---------------------------
# Processing Code
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

# Define model options (adjust file paths as needed)
model_options = {
    "MobileNetV2 (T2)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary2Model/MobileNetV2_model.h5",
    "VGG16 (T2)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary2Model/VGG16_model.h5",
    "DenseNet121 (T2)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary2Model/DenseNet121_model.h5",
    "V3_VGG16 (T3)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary3Model/VGG16_model.h5"
}

# Initially load the default model
selected_model = "VGG16 (T2)"
classifier = Classifier(model_options[selected_model])

offset = 45
imgSize = 250
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", 
          "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

mp_hands = mp.solutions.hands  # for landmark connections
hand_connections = mp_hands.HAND_CONNECTIONS

# ---------------------------
# Global variables for word typing
# ---------------------------
current_word = ""         # the final sentence built from letters and spaces
current_prediction = ""   # the letter predicted by the classifier

# Variables for delaying predictions and key presses
last_prediction_update_time = 0
prediction_update_delay = 0.05  # seconds delay for updating prediction
last_space_time = 0
space_cooldown = 0.5  # seconds delay between spacebar inputs

# ---------------------------
# Tkinter UI Setup
# ---------------------------
root = tk.Tk()
root.title("ASL - Fingerspelling Recognition")
root.geometry("1200x800")
root.configure(bg="#000000")  # overall dark background

# Left panel: Camera Feed (fixed)
left_frame = tk.Frame(root, width=800, height=800, bg="#000000")
left_frame.grid(row=0, column=0, sticky="nsew")
left_frame.grid_propagate(False)

# Right panel: Controls & Info (fixed)
# (Increased width to 750 to allow more space for the right-panel contents.)
right_frame = tk.Frame(root, width=715, height=800, bg="#1F1F1F")
right_frame.grid(row=0, column=1, sticky="nsew")
right_frame.grid_propagate(False)

# Configure grid weights for root to stretch panels
root.grid_columnconfigure(0, weight=2)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)

# --- Right Panel Layout ---
# We'll use a grid layout with rows for header, options, image displays, info, and buttons

# Header
header_label = tk.Label(right_frame, text="Sign Language Recognition", font=("Helvetica", 20, "bold"),
                        bg="#1F1F1F", fg="#00FFFF")
header_label.grid(row=0, column=0, columnspan=2, pady=(10, 5), sticky="ew")

# 1. Model Selection
model_label = tk.Label(right_frame, text="Select Model:", font=("Helvetica", 14),
                       bg="#1F1F1F", fg="#FFFFFF")
model_label.grid(row=1, column=0, padx=(10, 5), pady=5, sticky="w")

selected_model_var = tk.StringVar()
selected_model_var.set(selected_model)  # default value

def change_model(*args):
    global classifier
    sel = selected_model_var.get()
    new_path = model_options[sel]
    classifier = Classifier(new_path)
    history_text.configure(state='normal')
    history_text.insert(tk.END, f"Model changed to: {sel}\n")
    history_text.see(tk.END)
    history_text.configure(state='disabled')

model_option_menu = tk.OptionMenu(right_frame, selected_model_var, *model_options.keys(), command=change_model)
model_option_menu.config(font=("Helvetica", 14), bg="#333333", fg="#FFFFFF", highlightthickness=0)
model_option_menu["menu"].config(bg="#333333", fg="#FFFFFF")
model_option_menu.grid(row=1, column=1, padx=(5,10), pady=5, sticky="e")

# 2. Fixed Image Display Frames
# Set fixed dimensions (width x height)
FIXED_WIDTH = 280
FIXED_HEIGHT = 280

# Binary Image Display
binary_frame = tk.Frame(right_frame, width=FIXED_WIDTH, height=FIXED_HEIGHT, bg="#333333")
binary_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
binary_frame.grid_propagate(False)
binary_title = tk.Label(binary_frame, text="Binary Image", font=("Helvetica", 12),
                          bg="#333333", fg="#FFFFFF")
binary_title.pack(side="top", fill="x")
binary_label = tk.Label(binary_frame, bg="#333333")
binary_label.pack(expand=True, fill="both")

# Hand Landmarks Display
landmarks_frame = tk.Frame(right_frame, width=FIXED_WIDTH, height=FIXED_HEIGHT, bg="#333333")
landmarks_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
landmarks_frame.grid_propagate(False)
landmarks_title = tk.Label(landmarks_frame, text="Landmarks", font=("Helvetica", 12),
                            bg="#333333", fg="#FFFFFF")
landmarks_title.pack(side="top", fill="x")
landmarks_label = tk.Label(landmarks_frame, bg="#333333")
landmarks_label.pack(expand=True, fill="both")

# --- Set default black images in the binary and landmark panels ---
default_black = np.zeros((FIXED_HEIGHT, FIXED_WIDTH, 3), dtype=np.uint8)
default_black_pil = Image.fromarray(default_black)
default_black_tk = ImageTk.PhotoImage(image=default_black_pil)
binary_label.imgtk = default_black_tk
binary_label.configure(image=default_black_tk)
landmarks_label.imgtk = default_black_tk
landmarks_label.configure(image=default_black_tk)

# 3. Prediction Info
prediction_info_frame = tk.Frame(right_frame, bg="#1F1F1F")
prediction_info_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
prediction_info_frame.columnconfigure(0, weight=1)
prediction_info_frame.columnconfigure(1, weight=1)

word_label = tk.Label(prediction_info_frame, textvariable=tk.StringVar(), font=("Helvetica", 16),
                      bg="#1F1F1F", fg="#00FF00")
word_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
word_var = tk.StringVar()
word_var.set("")
word_label.config(textvariable=word_var)

prediction_label = tk.Label(prediction_info_frame, textvariable=tk.StringVar(), font=("Helvetica", 16),
                            bg="#1F1F1F", fg="#FF00FF")
prediction_label.grid(row=0, column=1, padx=5, pady=5, sticky="e")
prediction_var = tk.StringVar()
prediction_var.set("")
prediction_label.config(textvariable=prediction_var)

# 4. Buttons (Clear and Pronounce)
buttons_frame = tk.Frame(right_frame, bg="#1F1F1F")
buttons_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
buttons_frame.columnconfigure(0, weight=1)
buttons_frame.columnconfigure(1, weight=1)

def clear_word():
    global current_word
    current_word = ""
    history_text.configure(state='normal')
    history_text.insert(tk.END, "Clear pressed: Word cleared\n")
    history_text.see(tk.END)
    history_text.configure(state='disabled')
    word_var.set(current_word)

clear_button = tk.Button(buttons_frame, text="Clear", font=("Helvetica", 14),
                           bg="#333333", fg="#00FFFF", command=clear_word)
clear_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

def pronounce_sentence():
    global current_word
    engine = pyttsx3.init()
    sentence = current_word.strip()
    if sentence == "":
        history_text.configure(state='normal')
        history_text.insert(tk.END, "No sentence to pronounce.\n")
        history_text.see(tk.END)
        history_text.configure(state='disabled')
        return
    history_text.configure(state='normal')
    history_text.insert(tk.END, f"Pronouncing: {sentence}\n")
    history_text.see(tk.END)
    history_text.configure(state='disabled')
    engine.say(sentence)
    engine.runAndWait()

pronounce_button = tk.Button(buttons_frame, text="Pronounce", font=("Helvetica", 14),
                              bg="#333333", fg="#00FFFF", command=pronounce_sentence)
pronounce_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

# 5. History Text (at the bottom)
history_text = scrolledtext.ScrolledText(right_frame, width=40, height=8,
                                           bg="#2E2E2E", fg="#FFFFFF", font=("Helvetica", 12))
history_text.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
history_text.configure(state='disabled')

# ---------------------------
# Main Camera Feed Display (Left Panel)
# ---------------------------
main_image_label = tk.Label(left_frame, bg="#000000")
main_image_label.place(relwidth=1, relheight=1)  # fill entire left frame

# ---------------------------
# Key Binding Functions
# ---------------------------
def on_space_press(event):
    global current_word, current_prediction, last_space_time
    current_time = time.time()
    if current_time - last_space_time < space_cooldown:
        return  # Ignore if within cooldown
    last_space_time = current_time
    if current_prediction != "":
        current_word += current_prediction
        history_text.configure(state='normal')
        history_text.insert(tk.END, f"Entered: {current_prediction}\n")
        history_text.see(tk.END)
        history_text.configure(state='disabled')
    else:
        current_word += " "
        history_text.configure(state='normal')
        history_text.insert(tk.END, "Entered: [space]\n")
        history_text.see(tk.END)
        history_text.configure(state='disabled')
    word_var.set(current_word)

def on_backspace(event):
    global current_word
    if current_word:
        current_word = current_word[:-1]
        history_text.configure(state='normal')
        history_text.insert(tk.END, "Backspace pressed: Removed last character\n")
        history_text.see(tk.END)
        history_text.configure(state='disabled')
        word_var.set(current_word)

root.bind("<space>", on_space_press)
root.bind("<BackSpace>", on_backspace)

# ---------------------------
# Update Loop for Video Frames
# ---------------------------
def update_frame():
    global current_prediction, last_prediction_update_time
    success, img = cap.read()
    if not success:
        root.after(10, update_frame)
        return

    imgOutput = img.copy()
    hands, img = detector.findHands(img, draw=False)
    
    imgWhite_for_display = None
    imgCrop_landmarked_for_display = None

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]
        
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
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
            
            binaryMask = detect_skin(imgCrop)
            binary_result = np.zeros_like(imgCrop)
            binary_result[binaryMask > 0] = [255, 255, 255]
            
            if 'lmList' in hand:
                for lm in lm_list:
                    cv2.circle(binary_result, (lm[0] - x1, lm[1] - y1), 4, (0, 0, 0), -1)
                for connection in mp_hands.HAND_CONNECTIONS:
                    pt1 = (lm_list[connection[0]][0] - x1, lm_list[connection[0]][1] - y1)
                    pt2 = (lm_list[connection[1]][0] - x1, lm_list[connection[1]][1] - y1)
                    cv2.line(binary_result, pt1, pt2, (0, 0, 0), 2)
            
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
            
            imgWhiteRGB = cv2.cvtColor(imgWhite, cv2.COLOR_GRAY2BGR)
            prediction, index = classifier.getPrediction(imgWhiteRGB, draw=False)
            
            current_time = time.time()
            if current_time - last_prediction_update_time >= prediction_update_delay:
                if prediction[index] > 0.60 and 0 <= index < len(labels):
                    letter = labels[index]
                    current_prediction = letter
                    last_prediction_update_time = current_time
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                                  (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, letter, (x, y - 26),
                                cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset),
                                  (x + w + offset, y + h + offset), (255, 0, 255), 4)
                else:
                    current_prediction = ""
                    last_prediction_update_time = current_time
            
            prediction_var.set("Prediction: " + (current_prediction if current_prediction != "" else "[None]"))
            
            imgWhite_for_display = imgWhite.copy()
            imgCrop_landmarked_for_display = imgCrop_landmarked.copy()
    
    else:
        current_prediction = ""
        prediction_var.set("Prediction: [None]")
    
    imgOutput_rgb = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(imgOutput_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    main_image_label.imgtk = img_tk
    main_image_label.configure(image=img_tk)
    
    # Force images to be resized to the fixed dimensions
    if imgWhite_for_display is not None:
        imgWhite_rgb = cv2.cvtColor(imgWhite_for_display, cv2.COLOR_GRAY2RGB)
        imgWhite_pil = Image.fromarray(imgWhite_rgb).resize((FIXED_WIDTH, FIXED_HEIGHT), resample_method)
        imgWhite_tk = ImageTk.PhotoImage(image=imgWhite_pil)
        binary_label.imgtk = imgWhite_tk
        binary_label.configure(image=imgWhite_tk)
    
    if imgCrop_landmarked_for_display is not None:
        imgCrop_rgb = cv2.cvtColor(imgCrop_landmarked_for_display, cv2.COLOR_BGR2RGB)
        imgCrop_pil = Image.fromarray(imgCrop_rgb).resize((FIXED_WIDTH, FIXED_HEIGHT), resample_method)
        imgCrop_tk = ImageTk.PhotoImage(image=imgCrop_pil)
        landmarks_label.imgtk = imgCrop_tk
        landmarks_label.configure(image=imgCrop_tk)
    
    root.after(10, update_frame)

# ---------------------------
# Clean up on exit
# ---------------------------
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the update loop.
update_frame()
root.mainloop()



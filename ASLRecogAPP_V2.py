# import cv2
# import mediapipe as mp
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import tkinter as tk
# from tkinter import scrolledtext
# from PIL import Image, ImageTk
# import time  # for timing delays
# import pyttsx3  # for text-to-speech
# import os  # for creating directories and saving files

# # Determine the resampling method for resizing images
# try:
#     resample_method = Image.Resampling.LANCZOS
# except AttributeError:
#     resample_method = Image.ANTIALIAS

# # ---------------------------
# # Processing Code
# # ---------------------------
# def detect_skin(frame):
#     # Convert to YCrCb and equalize the luminance channel
#     ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
#     y_channel = ycrcb[:, :, 0]
#     y_eq = cv2.equalizeHist(y_channel)
#     ycrcb[:, :, 0] = y_eq

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

# # Initialize MediaPipe Selfie Segmentation (from your first code)
# mp_selfie_segmentation = mp.solutions.selfie_segmentation
# segmentation_module = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

# # Define model options (adjust file paths as needed)
# model_options = {
#     "MobileNetV2 (T2)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary2Model/MobileNetV2_model.h5",
#     "VGG16 (T2)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary2Model/VGG16_model.h5",
#     "DenseNet121 (T2)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary2Model/DenseNet121_model.h5",
#     "V3_VGG16 (T3)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary3Model/VGG16_model.h5",
#     "V4_VGG16 (T4)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary4Model/VGG16_model.h5",
#     "v4_MobileNetV2 (T4)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary4Model/MobileNetV2_model.h5",
#     "V5_MobileNetV2 (T5)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary5Model/MobileNetV2_model.h5",
#     "V5_VGG19 (T5)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary5Model/VGG19_model.h5"
# }

# # Initially load the default model
# selected_model = "VGG16 (T2)"
# classifier = Classifier(model_options[selected_model])

# offset = 45
# imgSize = 250
# labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", 
#           "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

# mp_hands = mp.solutions.hands  # for landmark connections
# hand_connections = mp_hands.HAND_CONNECTIONS

# # ---------------------------
# # Global variables for word typing
# # ---------------------------
# current_word = ""         # the final sentence built from letters and spaces
# current_prediction = ""   # the letter predicted by the classifier
# # Global variables for averaging predictions over 1 second
# prediction_sum = np.zeros(len(labels))
# prediction_count = 0
# last_avg_time = time.time()
# # Global variable for moving average buffer
# prediction_buffer = []  # Each element: (timestamp, prediction_array)

# # Variables for delaying predictions and key presses
# last_prediction_update_time = 0
# prediction_update_delay = 0.05  # seconds delay for updating prediction
# last_space_time = 0
# space_cooldown = 0.5  # seconds delay between spacebar inputs

# # ---------------------------
# # Global variables for FPS measurements
# # ---------------------------
# last_frame_time = time.time()  # For computing FPS between frames
# fps_values = []                # List to store fps values during the last 1 second
# frame_counter = 0              # Count total frames in the 1-second period
# pred_counter = 0               # Count frames with a prediction (hand detected)
# last_fps_report_time = time.time()  # Last time the FPS stats were computed

# # ---------------------------
# # Tkinter UI Setup
# # ---------------------------
# root = tk.Tk()
# root.title("ASL - Fingerspelling Recognition")
# # Adjust the overall window size to accommodate three panels
# root.geometry("1800x800")
# root.configure(bg="#000000")  # overall dark background

# # Configure grid weights for three columns
# root.grid_columnconfigure(0, weight=1)  # FPS panel
# root.grid_columnconfigure(1, weight=2)  # Main camera feed
# root.grid_columnconfigure(2, weight=1)  # Controls & Info
# root.grid_rowconfigure(0, weight=1)

# # --- Left Panel: Performance Metrics ---
# fps_frame = tk.Frame(root, width=300, height=800, bg="#444444")
# fps_frame.grid(row=0, column=0, sticky="nsew")
# fps_frame.grid_propagate(False)

# fps_frame_title = tk.Label(fps_frame, text="Performance Metrics", font=("Helvetica", 16, "bold"),
#                            bg="#444444", fg="#FFCC00")
# fps_frame_title.pack(pady=10)

# # A label to show which model is active
# max_fps_var = tk.StringVar()
# min_fps_var = tk.StringVar()
# avg_fps_var = tk.StringVar()
# pred_rate_var = tk.StringVar()

# # Set initial text values
# max_fps_var.set("Max FPS: 0")
# min_fps_var.set("Min FPS: 0")
# avg_fps_var.set("Avg FPS: 0")
# pred_rate_var.set("Predictions per frame: 0")

# model_fps_label = tk.Label(fps_frame, text="Model: " + selected_model, font=("Helvetica", 14),
#                            bg="#444444", fg="#00FF00")
# model_fps_label.pack(pady=5)

# max_fps_label = tk.Label(fps_frame, textvariable=max_fps_var, font=("Helvetica", 14),
#                          bg="#444444", fg="#FFFFFF")
# max_fps_label.pack(pady=5)

# min_fps_label = tk.Label(fps_frame, textvariable=min_fps_var, font=("Helvetica", 14),
#                          bg="#444444", fg="#FFFFFF")
# min_fps_label.pack(pady=5)

# avg_fps_label = tk.Label(fps_frame, textvariable=avg_fps_var, font=("Helvetica", 14),
#                          bg="#444444", fg="#FFFFFF")
# avg_fps_label.pack(pady=5)

# pred_rate_label = tk.Label(fps_frame, textvariable=pred_rate_var, font=("Helvetica", 14),
#                            bg="#444444", fg="#FFFFFF")
# pred_rate_label.pack(pady=5)

# # --- Center Panel: Main Camera Feed ---
# left_frame = tk.Frame(root, width=800, height=800, bg="#000000")
# left_frame.grid(row=0, column=1, sticky="nsew")
# left_frame.grid_propagate(False)

# main_image_label = tk.Label(left_frame, bg="#000000")
# main_image_label.place(relwidth=1, relheight=1)  # fill entire left frame

# # --- Right Panel: Controls & Info ---
# right_frame = tk.Frame(root, width=715, height=800, bg="#1F1F1F")
# right_frame.grid(row=0, column=2, sticky="nsew")
# right_frame.grid_propagate(False)

# # Header (Row 0)
# header_label = tk.Label(right_frame, text="ASL Fingerspelling Recognition", font=("Helvetica", 20, "bold"),
#                         bg="#1F1F1F", fg="#00FFFF")
# header_label.grid(row=0, column=0, columnspan=2, pady=(10, 5), sticky="ew")

# # 1. Model Selection (Row 1)
# model_label = tk.Label(right_frame, text="Select Model:", font=("Helvetica", 14),
#                        bg="#1F1F1F", fg="#FFFFFF")
# model_label.grid(row=1, column=0, padx=(10, 5), pady=5, sticky="w")

# selected_model_var = tk.StringVar()
# selected_model_var.set(selected_model)  # default value

# def change_model(*args):
#     global classifier
#     sel = selected_model_var.get()
#     new_path = model_options[sel]
#     classifier = Classifier(new_path)
#     # Update the model label in the FPS panel as well
#     model_fps_label.config(text="Model: " + sel)
#     history_text.configure(state='normal')
#     history_text.insert(tk.END, f"Model changed to: {sel}\n")
#     history_text.see(tk.END)
#     history_text.configure(state='disabled')

# model_option_menu = tk.OptionMenu(right_frame, selected_model_var, *model_options.keys(), command=change_model)
# model_option_menu.config(font=("Helvetica", 14), bg="#333333", fg="#FFFFFF", highlightthickness=0)
# model_option_menu["menu"].config(bg="#333333", fg="#FFFFFF")
# model_option_menu.grid(row=1, column=1, padx=(5,10), pady=5, sticky="e")

# # 1.5. Segmentation Filter Option (New Row 2)
# segmentation_enabled_var = tk.BooleanVar()
# segmentation_enabled_var.set(False)  # default is off
# segmentation_check = tk.Checkbutton(right_frame, text="Enable Segmentation Filter", variable=segmentation_enabled_var,
#                                     font=("Helvetica", 14), bg="#1F1F1F", fg="#FFFFFF",
#                                     activebackground="#1F1F1F", activeforeground="#00FFFF", selectcolor="#00FF00")
# segmentation_check.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="w")

# # 2. Fixed Image Display Frames (Now starting at Row 3)
# FIXED_WIDTH = 280
# FIXED_HEIGHT = 280

# # Binary Image Display
# binary_frame = tk.Frame(right_frame, width=FIXED_WIDTH, height=FIXED_HEIGHT, bg="#333333")
# binary_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
# binary_frame.grid_propagate(False)
# binary_title = tk.Label(binary_frame, text="Binary Image", font=("Helvetica", 12),
#                           bg="#333333", fg="#FFFFFF")
# binary_title.pack(side="top", fill="x")
# binary_label = tk.Label(binary_frame, bg="#333333")
# binary_label.pack(expand=True, fill="both")

# # Hand Landmarks Display
# landmarks_frame = tk.Frame(right_frame, width=FIXED_WIDTH, height=FIXED_HEIGHT, bg="#333333")
# landmarks_frame.grid(row=3, column=1, padx=10, pady=10, sticky="nsew")
# landmarks_frame.grid_propagate(False)
# landmarks_title = tk.Label(landmarks_frame, text="Landmarks", font=("Helvetica", 12),
#                             bg="#333333", fg="#FFFFFF")
# landmarks_title.pack(side="top", fill="x")
# landmarks_label = tk.Label(landmarks_frame, bg="#333333")
# landmarks_label.pack(expand=True, fill="both")

# # Set default black images in the binary and landmark panels
# default_black = np.zeros((FIXED_HEIGHT, FIXED_WIDTH, 3), dtype=np.uint8)
# default_black_pil = Image.fromarray(default_black)
# default_black_tk = ImageTk.PhotoImage(image=default_black_pil)
# binary_label.imgtk = default_black_tk
# binary_label.configure(image=default_black_tk)
# landmarks_label.imgtk = default_black_tk
# landmarks_label.configure(image=default_black_tk)

# # 3. Prediction Info (Now Row 4)
# prediction_info_frame = tk.Frame(right_frame, bg="#1F1F1F")
# prediction_info_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
# prediction_info_frame.columnconfigure(0, weight=1)
# prediction_info_frame.columnconfigure(1, weight=1)

# word_label = tk.Label(prediction_info_frame, textvariable=tk.StringVar(), font=("Helvetica", 16),
#                       bg="#1F1F1F", fg="#00FF00")
# word_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
# word_var = tk.StringVar()
# word_var.set("")
# word_label.config(textvariable=word_var)

# prediction_label = tk.Label(prediction_info_frame, textvariable=tk.StringVar(), font=("Helvetica", 16),
#                             bg="#1F1F1F", fg="#FF00FF")
# prediction_label.grid(row=0, column=1, padx=5, pady=5, sticky="e")
# prediction_var = tk.StringVar()
# prediction_var.set("")
# prediction_label.config(textvariable=prediction_var)

# # 4. Buttons (Now Row 5)
# buttons_frame = tk.Frame(right_frame, bg="#1F1F1F")
# buttons_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
# buttons_frame.columnconfigure(0, weight=1)
# buttons_frame.columnconfigure(1, weight=1)

# def clear_word():
#     global current_word
#     current_word = ""
#     history_text.configure(state='normal')
#     history_text.insert(tk.END, "Clear pressed: Word cleared\n")
#     history_text.see(tk.END)
#     history_text.configure(state='disabled')
#     word_var.set(current_word)

# clear_button = tk.Button(buttons_frame, text="Clear", font=("Helvetica", 14),
#                            bg="#333333", fg="#00FFFF", command=clear_word)
# clear_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

# def pronounce_sentence():
#     global current_word
#     engine = pyttsx3.init()
#     sentence = current_word.strip()
#     if sentence == "":
#         history_text.configure(state='normal')
#         history_text.insert(tk.END, "No sentence to pronounce.\n")
#         history_text.see(tk.END)
#         history_text.configure(state='disabled')
#         return
#     history_text.configure(state='normal')
#     history_text.insert(tk.END, f"Pronouncing: {sentence}\n")
#     history_text.see(tk.END)
#     history_text.configure(state='disabled')
#     engine.say(sentence)
#     engine.runAndWait()

# pronounce_button = tk.Button(buttons_frame, text="Pronounce", font=("Helvetica", 14),
#                               bg="#333333", fg="#00FFFF", command=pronounce_sentence)
# pronounce_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

# # 5. History Text (Now Row 6)
# history_text = scrolledtext.ScrolledText(right_frame, width=40, height=8,
#                                            bg="#2E2E2E", fg="#FFFFFF", font=("Helvetica", 12))
# history_text.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
# history_text.configure(state='disabled')

# instructions_label = tk.Label(
#     right_frame,
#     text="1. Enter word: Space Bar  |  2. Delete last character: Backspace",
#     font=("Helvetica", 10),
#     bg="#1F1F1F",
#     fg="#AAAAAA"
# )
# # 6. Instructions (Now Row 7)
# instructions_label.grid(row=7, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")

# # ---------------------------
# # Key Binding Functions
# # ---------------------------
# def on_space_press(event):
#     global current_word, current_prediction, last_space_time
#     current_time = time.time()
#     if current_time - last_space_time < space_cooldown:
#         return  # Ignore if within cooldown
#     last_space_time = current_time
#     if current_prediction != "":
#         current_word += current_prediction
#         history_text.configure(state='normal')
#         history_text.insert(tk.END, f"Entered: {current_prediction}\n")
#         history_text.see(tk.END)
#         history_text.configure(state='disabled')
#     else:
#         current_word += " "
#         history_text.configure(state='normal')
#         history_text.insert(tk.END, "Entered: [space]\n")
#         history_text.see(tk.END)
#         history_text.configure(state='disabled')
#     word_var.set(current_word)

# def on_backspace(event):
#     global current_word
#     if current_word:
#         current_word = current_word[:-1]
#         history_text.configure(state='normal')
#         history_text.insert(tk.END, "Backspace pressed: Removed last character\n")
#         history_text.see(tk.END)
#         history_text.configure(state='disabled')
#         word_var.set(current_word)

# root.bind("<space>", on_space_press)
# root.bind("<BackSpace>", on_backspace)

# # ---------------------------
# # Main update_frame() function
# # ---------------------------
# def update_frame():
#     global current_prediction, last_prediction_update_time, prediction_buffer
#     global last_frame_time, fps_values, frame_counter, pred_counter, last_fps_report_time

#     success, img = cap.read()
#     if not success:
#         root.after(10, update_frame)
#         return

#     # ---------------------------
#     # FPS Measurement (per frame)
#     # ---------------------------
#     current_time = time.time()
#     dt = current_time - last_frame_time
#     if dt > 0:
#         current_fps = 1.0 / dt
#     else:
#         current_fps = 0
#     last_frame_time = current_time

#     fps_values.append(current_fps)
#     frame_counter += 1

#     imgOutput = img.copy()
#     hands, img = detector.findHands(img, draw=False)

#     # If a hand is detected, count this frame as having a prediction.
#     if hands:
#         pred_counter += 1

#     imgWhite_for_display = None
#     imgCrop_landmarked_for_display = None

#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
#         y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
#         x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
#         imgCrop = img[y1:y2, x1:x2]

#         # If segmentation is enabled, apply the segmentation filter to the cropped hand image.
#         if segmentation_enabled_var.get():
#             rgb_crop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)
#             seg_results = segmentation_module.process(rgb_crop)
#             seg_mask = seg_results.segmentation_mask
#             threshold_seg = 0.5  # Adjust threshold if needed
#             mask_binary_seg = (seg_mask > threshold_seg).astype(np.uint8) * 255
#             mask_binary_seg = cv2.cvtColor(mask_binary_seg, cv2.COLOR_GRAY2BGR)
#             imgCrop = cv2.bitwise_and(imgCrop, mask_binary_seg)

#         if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
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

#             binaryMask = detect_skin(imgCrop)
#             binary_result = np.zeros_like(imgCrop)
#             binary_result[binaryMask > 0] = [255, 255, 255]

#             if 'lmList' in hand:
#                 for lm in lm_list:
#                     cv2.circle(binary_result, (lm[0] - x1, lm[1] - y1), 4, (0, 0, 0), -1)
#                 for connection in mp_hands.HAND_CONNECTIONS:
#                     pt1 = (lm_list[connection[0]][0] - x1, lm_list[connection[0]][1] - y1)
#                     pt2 = (lm_list[connection[1]][0] - x1, lm_list[connection[1]][1] - y1)
#                     cv2.line(binary_result, pt1, pt2, (0, 0, 0), 2)

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
#                 hCal = int(k * h)
#                 imgResize = cv2.resize(binary_result, (imgSize, hCal))
#                 hGap = (imgSize - hCal) // 2
#                 imgWhite[hGap:hGap + hCal, :] = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)

#             imgWhiteRGB = cv2.cvtColor(imgWhite, cv2.COLOR_GRAY2BGR)
#             prediction, index = classifier.getPrediction(imgWhiteRGB, draw=False)

#             # --- Moving Average over last 1 second ---
#             prediction_buffer.append((current_time, prediction))
#             prediction_buffer[:] = [(t, p) for (t, p) in prediction_buffer if current_time - t <= 1.0]
#             if len(prediction_buffer) > 0:
#                 avg_prediction = np.mean([p for (t, p) in prediction_buffer], axis=0)
#                 best_index = np.argmax(avg_prediction)
#                 best_letter = labels[best_index]
#                 best_prob = avg_prediction[best_index]
#                 current_prediction = best_letter

#                 # Draw a continuously visible pink box around the hand.
#                 box_x = x - offset
#                 box_y = y - offset - 50
#                 box_width = 150  # Adjust width as needed
#                 box_height = 50
#                 cv2.rectangle(imgOutput, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 0, 255), cv2.FILLED)
#                 text = f"{best_letter}: {best_prob:.2f}"
#                 cv2.putText(imgOutput, text, (box_x + 5, box_y + 35), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
#                 cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
#             else:
#                 current_prediction = ""
            
#             prediction_var.set("Prediction: " + (current_prediction if current_prediction != "" else "[None]"))
            
#             imgWhite_for_display = imgWhite.copy()
#             imgCrop_landmarked_for_display = imgCrop_landmarked.copy()
    
#     else:
#         current_prediction = ""
#         prediction_var.set("Prediction: [None]")

#     imgOutput_rgb = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
#     img_pil = Image.fromarray(imgOutput_rgb)
#     img_tk = ImageTk.PhotoImage(image=img_pil)
#     main_image_label.imgtk = img_tk
#     main_image_label.configure(image=img_tk)
    
#     # Force images to be resized to the fixed dimensions for right-panel displays.
#     if imgWhite_for_display is not None:
#         imgWhite_rgb = cv2.cvtColor(imgWhite_for_display, cv2.COLOR_GRAY2RGB)
#         imgWhite_pil = Image.fromarray(imgWhite_rgb).resize((FIXED_WIDTH, FIXED_HEIGHT), resample_method)
#         imgWhite_tk = ImageTk.PhotoImage(image=imgWhite_pil)
#         binary_label.imgtk = imgWhite_tk
#         binary_label.configure(image=imgWhite_tk)
    
#     if imgCrop_landmarked_for_display is not None:
#         imgCrop_rgb = cv2.cvtColor(imgCrop_landmarked_for_display, cv2.COLOR_BGR2RGB)
#         imgCrop_pil = Image.fromarray(imgCrop_rgb).resize((FIXED_WIDTH, FIXED_HEIGHT), resample_method)
#         imgCrop_tk = ImageTk.PhotoImage(image=imgCrop_pil)
#         landmarks_label.imgtk = imgCrop_tk
#         landmarks_label.configure(image=imgCrop_tk)
    
#     # ---------------------------
#     # Update FPS Stats Every Second
#     # ---------------------------
#     if current_time - last_fps_report_time >= 1.0:
#         # Compute max, min, and average FPS over the last second
#         max_fps = max(fps_values) if fps_values else 0
#         min_fps = min(fps_values) if fps_values else 0
#         avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0
#         # Compute average predictions per frame (0 or 1 per frame)
#         pred_rate = (pred_counter / frame_counter) if frame_counter > 0 else 0

#         max_fps_var.set(f"Max FPS: {max_fps:.2f}")
#         min_fps_var.set(f"Min FPS: {min_fps:.2f}")
#         avg_fps_var.set(f"Avg FPS: {avg_fps:.2f}")
#         pred_rate_var.set(f"Predictions per frame: {pred_rate:.2f}")

#         # Log the FPS and prediction stats to a text file for the current model
#         log_folder = "AppFPS_Results"
#         os.makedirs(log_folder, exist_ok=True)
#         log_filename = os.path.join(log_folder, f"{selected_model_var.get()}_results.txt")
#         with open(log_filename, "a") as log_file:
#             log_line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Max FPS: {max_fps:.2f}, Min FPS: {min_fps:.2f}, " \
#                        f"Avg FPS: {avg_fps:.2f}, Predictions per frame: {pred_rate:.2f}\n"
#             log_file.write(log_line)
        
#         # Reset the counters and list for the next one-second period
#         fps_values = []
#         frame_counter = 0
#         pred_counter = 0
#         last_fps_report_time = current_time

#     root.after(10, update_frame)

# # ---------------------------
# # Clean up on exit
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
import time  # for timing delays
import pyttsx3  # for text-to-speech
import os  # for creating directories and saving files
import psutil  # for resource usage info
import platform
try:
    import GPUtil
except ImportError:
    GPUtil = None

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

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation_module = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

# Define model options (adjust file paths as needed)
model_options = {
    "MobileNetV2 (T2)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary2Model/MobileNetV2_model.h5",
    "VGG16 (T2)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary2Model/VGG16_model.h5",
    "DenseNet121 (T2)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary2Model/DenseNet121_model.h5",
    "V3_VGG16 (T3)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary3Model/VGG16_model.h5",
    "V4_VGG16 (T4)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary4Model/VGG16_model.h5",
    "v4_MobileNetV2 (T4)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary4Model/MobileNetV2_model.h5",
    "V5_MobileNetV2 (T5)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary5Model/MobileNetV2_model.h5",
    "V5_VGG19 (T5)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinary5Model/VGG19_model.h5"
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
current_word = ""
current_prediction = ""
prediction_buffer = []  # Each element: (timestamp, prediction_array)

last_prediction_update_time = 0
prediction_update_delay = 0.05
last_space_time = 0
space_cooldown = 0.5

# ---------------------------
# Global variables for FPS measurements
# ---------------------------
last_frame_time = time.time()
fps_values = []
frame_counter = 0
pred_counter = 0
last_fps_report_time = time.time()

# ---------------------------
# Tkinter UI Setup
# ---------------------------
root = tk.Tk()
root.title("ASL - Fingerspelling Recognition")
# Adjust overall window size to accommodate three panels; expanded left panel.
root.geometry("2100x800")
root.configure(bg="#000000")

# Configure grid weights for three columns:
# Column 0: Performance & resource metrics (expanded width)
# Column 1: Main camera feed
# Column 2: Controls & Info
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=2)
root.grid_columnconfigure(2, weight=1)
root.grid_rowconfigure(0, weight=1)

# --- Left Panel: Performance Metrics and Resource Info ---
fps_frame = tk.Frame(root, width=500, height=800, bg="#444444")
fps_frame.grid(row=0, column=0, sticky="nsew")
fps_frame.grid_propagate(False)

fps_frame_title = tk.Label(fps_frame, text="Performance Metrics", font=("Helvetica", 16, "bold"),
                           bg="#444444", fg="#00FFFF")
fps_frame_title.pack(pady=10)

# FPS info labels
max_fps_var = tk.StringVar(value="Max FPS: 0")
min_fps_var = tk.StringVar(value="Min FPS: 0")
avg_fps_var = tk.StringVar(value="Avg FPS: 0")
pred_rate_var = tk.StringVar(value="Predictions per frame: 0")
# New field for Predictions Per Second (PPS)
pps_var = tk.StringVar(value="PPS: 0")

model_fps_label = tk.Label(fps_frame, text="Model: " + selected_model, font=("Helvetica", 14),
                           bg="#444444", fg="#00FF00")
model_fps_label.pack(pady=5)

max_fps_label = tk.Label(fps_frame, textvariable=max_fps_var, font=("Helvetica", 14),
                         bg="#444444", fg="#FFFFFF")
max_fps_label.pack(pady=5)

min_fps_label = tk.Label(fps_frame, textvariable=min_fps_var, font=("Helvetica", 14),
                         bg="#444444", fg="#FFFFFF")
min_fps_label.pack(pady=5)

avg_fps_label = tk.Label(fps_frame, textvariable=avg_fps_var, font=("Helvetica", 14),
                         bg="#444444", fg="#FFFFFF")
avg_fps_label.pack(pady=5)

pred_rate_label = tk.Label(fps_frame, textvariable=pred_rate_var, font=("Helvetica", 14),
                           bg="#444444", fg="#FFFFFF")
pred_rate_label.pack(pady=5)

# New PPS label
pps_label = tk.Label(fps_frame, textvariable=pps_var, font=("Helvetica", 14),
                     bg="#444444", fg="#FFFFFF")
pps_label.pack(pady=5)

# --- Resource Info Section ---
resource_frame = tk.Frame(fps_frame, bg="#444444")
resource_frame.pack(pady=20)

resource_title = tk.Label(resource_frame, text="Resource Info", font=("Helvetica", 16, "bold"),
                          bg="#444444", fg="#FFCC00")
resource_title.pack(pady=5)

cpu_info_var = tk.StringVar(value="CPU: Loading...")
gpu_info_var = tk.StringVar(value="GPU: Loading...")
memory_info_var = tk.StringVar(value="Memory: Loading...")

cpu_info_label = tk.Label(resource_frame, textvariable=cpu_info_var, font=("Helvetica", 12),
                          bg="#444444", fg="#FFFFFF")
cpu_info_label.pack(pady=2)

gpu_info_label = tk.Label(resource_frame, textvariable=gpu_info_var, font=("Helvetica", 12),
                          bg="#444444", fg="#FFFFFF")
gpu_info_label.pack(pady=2)

memory_info_label = tk.Label(resource_frame, textvariable=memory_info_var, font=("Helvetica", 12),
                             bg="#444444", fg="#FFFFFF")
memory_info_label.pack(pady=2)

def update_resource_info():
    # CPU Info
    cpu_model = platform.processor()
    if not cpu_model:
        cpu_model = "Unknown CPU"
    cores = psutil.cpu_count(logical=True)
    cpu_info_var.set(f"CPU: {cpu_model} ({cores} cores)")
    
    # GPU Info
    if GPUtil is not None:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_names = ", ".join([gpu.name for gpu in gpus])
            gpu_info_var.set("GPU: " + gpu_names)
        else:
            gpu_info_var.set("GPU: None")
    else:
        gpu_info_var.set("GPU: Not available")
    
    # Memory Info
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    available_gb = mem.available / (1024**3)
    memory_info_var.set(f"Memory: {available_gb:.1f}GB free / {total_gb:.1f}GB")
    
    root.after(1000, update_resource_info)

update_resource_info()

# --- Center Panel: Main Camera Feed ---
left_frame = tk.Frame(root, width=800, height=800, bg="#000000")
left_frame.grid(row=0, column=1, sticky="nsew")
left_frame.grid_propagate(False)

main_image_label = tk.Label(left_frame, bg="#000000")
main_image_label.place(relwidth=1, relheight=1)

# --- Right Panel: Controls & Info ---
right_frame = tk.Frame(root, width=550, height=800, bg="#1F1F1F")
right_frame.grid(row=0, column=2, sticky="nsew")
right_frame.grid_propagate(False)

# Configure the right frame’s grid to have two equal columns
right_frame.grid_columnconfigure(0, weight=1)
right_frame.grid_columnconfigure(1, weight=1)

# Header (Row 0)
header_label = tk.Label(right_frame, text="ASL Fingerspelling Recognition", font=("Helvetica", 20, "bold"),
                        bg="#1F1F1F", fg="#00FFFF", anchor="center")
header_label.grid(row=0, column=0, columnspan=2, pady=(10, 5), sticky="ew")

# 1. Model Selection (Row 1)
model_label = tk.Label(right_frame, text="Select Model:", font=("Helvetica", 14),
                       bg="#1F1F1F", fg="#FFFFFF", anchor="e")
model_label.grid(row=1, column=0, padx=(10, 5), pady=5, sticky="e")

selected_model_var = tk.StringVar(value=selected_model)

def change_model(*args):
    global classifier
    sel = selected_model_var.get()
    new_path = model_options[sel]
    classifier = Classifier(new_path)
    model_fps_label.config(text="Model: " + sel)
    history_text.configure(state='normal')
    history_text.insert(tk.END, f"Model changed to: {sel}\n")
    history_text.see(tk.END)
    history_text.configure(state='disabled')

model_option_menu = tk.OptionMenu(right_frame, selected_model_var, *model_options.keys(), command=change_model)
model_option_menu.config(font=("Helvetica", 14), bg="#333333", fg="#FFFFFF", highlightthickness=0)
model_option_menu["menu"].config(bg="#333333", fg="#FFFFFF")
model_option_menu.grid(row=1, column=1, padx=(5,10), pady=5, sticky="w")

# 1.5. Segmentation Filter Option (Row 2)
segmentation_enabled_var = tk.BooleanVar(value=False)
segmentation_check = tk.Checkbutton(right_frame, text="Enable Segmentation Filter", variable=segmentation_enabled_var,
                                    font=("Helvetica", 14), bg="#1F1F1F", fg="#FFFFFF",
                                    activebackground="#1F1F1F", activeforeground="#00FFFF", selectcolor="#000000")
segmentation_check.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

# 2. Fixed Image Display Frames (Row 3)
FIXED_WIDTH = 280
FIXED_HEIGHT = 280

# Binary Image Display
binary_frame = tk.Frame(right_frame, width=FIXED_WIDTH, height=FIXED_HEIGHT, bg="#333333")
binary_frame.grid(row=3, column=0, padx=10, pady=10, sticky="n")
binary_frame.grid_propagate(False)
binary_title = tk.Label(binary_frame, text="Binary Image", font=("Helvetica", 12),
                          bg="#333333", fg="#FFFFFF")
binary_title.pack(side="top", fill="x")
binary_label = tk.Label(binary_frame, bg="#333333")
binary_label.pack(expand=True, fill="both")

# Hand Landmarks Display
landmarks_frame = tk.Frame(right_frame, width=FIXED_WIDTH, height=FIXED_HEIGHT, bg="#333333")
landmarks_frame.grid(row=3, column=1, padx=10, pady=10, sticky="n")
landmarks_frame.grid_propagate(False)
landmarks_title = tk.Label(landmarks_frame, text="Landmarks", font=("Helvetica", 12),
                            bg="#333333", fg="#FFFFFF")
landmarks_title.pack(side="top", fill="x")
landmarks_label = tk.Label(landmarks_frame, bg="#333333")
landmarks_label.pack(expand=True, fill="both")

default_black = np.zeros((FIXED_HEIGHT, FIXED_WIDTH, 3), dtype=np.uint8)
default_black_pil = Image.fromarray(default_black)
default_black_tk = ImageTk.PhotoImage(image=default_black_pil)
binary_label.imgtk = default_black_tk
binary_label.configure(image=default_black_tk)
landmarks_label.imgtk = default_black_tk
landmarks_label.configure(image=default_black_tk)

# 3. Prediction Info (Row 4)
prediction_info_frame = tk.Frame(right_frame, bg="#1F1F1F")
prediction_info_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
# Configure columns for spacing within prediction_info_frame
prediction_info_frame.grid_columnconfigure(0, weight=1)
prediction_info_frame.grid_columnconfigure(1, weight=1)

word_label = tk.Label(prediction_info_frame, text="", font=("Helvetica", 16),
                      bg="#1F1F1F", fg="#00FF00", anchor="e")
word_label.grid(row=0, column=0, padx=(5,20), pady=5, sticky="e")
word_var = tk.StringVar(value="")
word_label.config(textvariable=word_var)

prediction_label = tk.Label(prediction_info_frame, text="", font=("Helvetica", 16),
                            bg="#1F1F1F", fg="#FF00FF", anchor="w")
prediction_label.grid(row=0, column=1, padx=(20,5), pady=5, sticky="w")
prediction_var = tk.StringVar(value="")
prediction_label.config(textvariable=prediction_var)

# 4. Buttons (Row 5)
buttons_frame = tk.Frame(right_frame, bg="#1F1F1F")
buttons_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
# Configure columns in buttons_frame so that buttons have equal width
buttons_frame.grid_columnconfigure(0, weight=1)
buttons_frame.grid_columnconfigure(1, weight=1)

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

# 5. History Text (Row 6)
history_text = scrolledtext.ScrolledText(right_frame, width=40, height=8,
                                           bg="#2E2E2E", fg="#FFFFFF", font=("Helvetica", 12))
history_text.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
history_text.configure(state='disabled')

instructions_label = tk.Label(
    right_frame,
    text="1. Enter word: Space Bar  |  2. Delete last character: Backspace",
    font=("Helvetica", 10),
    bg="#1F1F1F",
    fg="#AAAAAA"
)
instructions_label.grid(row=7, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")

# ---------------------------
# Key Binding Functions
# ---------------------------
def on_space_press(event):
    global current_word, current_prediction, last_space_time
    current_time = time.time()
    if current_time - last_space_time < space_cooldown:
        return
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
# Main update_frame() function
# ---------------------------
def update_frame():
    global current_prediction, last_frame_time, fps_values, frame_counter, pred_counter, last_fps_report_time

    success, img = cap.read()
    if not success:
        root.after(10, update_frame)
        return

    # FPS Measurement (per frame)
    current_time = time.time()
    dt = current_time - last_frame_time
    current_fps = 1.0 / dt if dt > 0 else 0
    last_frame_time = current_time

    fps_values.append(current_fps)
    frame_counter += 1

    imgOutput = img.copy()
    hands, img = detector.findHands(img, draw=False)

    if hands:
        pred_counter += 1

    imgWhite_for_display = None
    imgCrop_landmarked_for_display = None

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if segmentation_enabled_var.get():
            rgb_crop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)
            seg_results = segmentation_module.process(rgb_crop)
            seg_mask = seg_results.segmentation_mask
            threshold_seg = 0.5
            mask_binary_seg = (seg_mask > threshold_seg).astype(np.uint8) * 255
            mask_binary_seg = cv2.cvtColor(mask_binary_seg, cv2.COLOR_GRAY2BGR)
            imgCrop = cv2.bitwise_and(imgCrop, mask_binary_seg)

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
                hCal = int(k * h)
                imgResize = cv2.resize(binary_result, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hGap + hCal, :] = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)

            imgWhiteRGB = cv2.cvtColor(imgWhite, cv2.COLOR_GRAY2BGR)
            prediction, index = classifier.getPrediction(imgWhiteRGB, draw=False)

            prediction_buffer.append((current_time, prediction))
            prediction_buffer[:] = [(t, p) for (t, p) in prediction_buffer if current_time - t <= 1.0]
            if prediction_buffer:
                avg_prediction = np.mean([p for (t, p) in prediction_buffer], axis=0)
                best_index = np.argmax(avg_prediction)
                best_letter = labels[best_index]
                best_prob = avg_prediction[best_index]
                current_prediction = best_letter

                box_x = x - offset
                box_y = y - offset - 50
                box_width = 150
                box_height = 50
                cv2.rectangle(imgOutput, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 0, 255), cv2.FILLED)
                text = f"{best_letter}: {best_prob:.2f}"
                cv2.putText(imgOutput, text, (box_x + 5, box_y + 35), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
            else:
                current_prediction = ""
            
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
    
    if current_time - last_fps_report_time >= 1.0:
        max_fps = max(fps_values) if fps_values else 0
        min_fps = min(fps_values) if fps_values else 0
        avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0
        pred_rate = (pred_counter / frame_counter) if frame_counter > 0 else 0
        # Compute Predictions Per Second (PPS)
        pps = pred_counter

        max_fps_var.set(f"Max FPS: {max_fps:.2f}")
        min_fps_var.set(f"Min FPS: {min_fps:.2f}")
        avg_fps_var.set(f"Avg FPS: {avg_fps:.2f}")
        pred_rate_var.set(f"Predictions per frame: {pred_rate:.2f}")
        pps_var.set(f"PPS: {pps}")

        # Log FPS and prediction stats; also log device info at the very top if file is new/empty.
        log_folder = "AppFPS_Results"
        os.makedirs(log_folder, exist_ok=True)
        log_filename = os.path.join(log_folder, f"{selected_model_var.get()}_results.txt")
        if not os.path.exists(log_filename) or os.stat(log_filename).st_size == 0:
            with open(log_filename, "w") as log_file:
                cpu_model = platform.processor() or "Unknown CPU"
                cores = psutil.cpu_count(logical=True)
                cpu_info_str = f"CPU: {cpu_model} ({cores} cores)"
                if GPUtil is not None:
                    gpus = GPUtil.getGPUs()
                    gpu_info_str = "GPU: " + (", ".join([gpu.name for gpu in gpus]) if gpus else "None")
                else:
                    gpu_info_str = "GPU: Not available"
                mem = psutil.virtual_memory()
                total_gb = mem.total / (1024**3)
                available_gb = mem.available / (1024**3)
                memory_info_str = f"Memory: {available_gb:.1f}GB free / {total_gb:.1f}GB total"
                log_file.write("Device Info:\n")
                log_file.write(cpu_info_str + "\n")
                log_file.write(gpu_info_str + "\n")
                log_file.write(memory_info_str + "\n")
                log_file.write("----------\n")
        with open(log_filename, "a") as log_file:
            log_line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Max FPS: {max_fps:.2f}, "
                        f"Min FPS: {min_fps:.2f}, Avg FPS: {avg_fps:.2f}, "
                        f"Predictions per frame: {pred_rate:.2f}, PPS: {pps}\n")
            log_file.write(log_line)
        
        fps_values = []
        frame_counter = 0
        pred_counter = 0
        last_fps_report_time = current_time

    root.after(10, update_frame)

# ---------------------------
# Clean up on exit
# ---------------------------
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

update_frame()
root.mainloop()


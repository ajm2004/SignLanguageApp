import cv2 
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tkinter as tk
from tkinter import scrolledtext, messagebox
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
# Ensemble Classifier Definition
# ---------------------------
class EnsembleClassifier:
    def __init__(self, model_paths):
        # Create a list of classifiers for each selected model path.
        self.classifiers = [Classifier(path) for path in model_paths]

    def getPrediction(self, img, draw=False):
        predictions = []
        # Get each classifier's prediction
        for clf in self.classifiers:
            pred, idx = clf.getPrediction(img, draw)
            predictions.append(pred)
        # Average the predictions (assuming they are probability arrays)
        avg_pred = np.mean(predictions, axis=0)
        best_index = np.argmax(avg_pred)
        return avg_pred, best_index

# ---------------------------
# Processing Code (unchanged)
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
# Initialization (unchanged)
# ---------------------------
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation_module = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

# Define model options (adjust file paths as needed)
model_options = {
    "MobileNetV2 (T2)": "TrainedBinaryNewModel/MobileNetV2_model.h5",
    "VGG16 (T2)": "TrainedBinaryNewModel/VGG16_model.h5",
    "DenseNet121 (T2)": "TrainedBinaryNewModel/DenseNet121_model.h5",
    "VGG19 (T2)": "TrainedBinaryNewModel/VGG19_model.h5",
    "Fusion Model (T2)": "TrainedBinaryNewModel/FusionModel_model.h5",
    "MobileNet (T2)": "TrainedBinaryNewModel/MobileNet_model.h5",
    # "CNN Model (T2)": "C:/Users/User/OneDrive/Documents/SignLanguageApp/TrainedBinaryNewModel/BasicCNN_model.h5",
    "NASNetMobile (T2)": "TrainedBinaryNewModel/NASNetMobile_model.h5",
    "MultiScaleAttentionalCNN (T2)": "TrainedBinaryNewModel/MultiScaleAttentionalCNN_model.h5",
}

# Initially load the default model
selected_model = "MobileNet (T2)"
classifier = Classifier(model_options[selected_model])

offset = 45
imgSize = 250
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "HI", "I", "J", "K", "L", 
          "M", "N", "O", "P", "Q", "R", "S", " ", "T", "U", "V", "W", "X", "Y", "Z"]

mp_hands = mp.solutions.hands  # for landmark connections
hand_connections = mp_hands.HAND_CONNECTIONS

# ---------------------------
# Globals for word typing and auto type (unchanged)
# ---------------------------
current_word = ""
current_prediction = ""
prediction_buffer = []  # Each element: (timestamp, prediction_array)

last_prediction_update_time = 0
prediction_update_delay = 0.05
last_space_time = 0
space_cooldown = 0.5

# Globals for FPS measurements
last_frame_time = time.time()
fps_values = []
frame_counter = 0
pred_counter = 0
last_fps_report_time = time.time()

# Globals for Auto Type functionality
auto_type_start_time = None
auto_type_last_prediction = None



# ---------------------------
# Tkinter UI Setup (Improved)
# ---------------------------
root = tk.Tk()
root.title("ASL - Fingerspelling Recognition")
# Increase overall window height to accommodate additional content
root.geometry("2100x970")
root.configure(bg="#1e1e1e")

# ---------------------------
# Global variables for Pronunciation Accent (pyttsx3)
# ---------------------------
tts_engine = pyttsx3.init()
available_voices = tts_engine.getProperty('voices')
voice_dict = {}
for voice in available_voices:
    # Use the voice name as key (you can modify to include language info if needed)
    voice_dict[voice.name] = voice.id
default_voice_name = list(voice_dict.keys())[0] if voice_dict else ""
selected_voice_var = tk.StringVar(value=default_voice_name)

# Configure grid layout for the root window
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=0)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=0)

# ---------------------------
# Menubar (New)
# ---------------------------
menubar = tk.Menu(root, bg="#2c2f33", fg="#ffffff")
# File Menu
file_menu = tk.Menu(menubar, tearoff=0, bg="#2c2f33", fg="#ffffff")
file_menu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=file_menu)
# Help Menu
help_menu = tk.Menu(menubar, tearoff=0, bg="#2c2f33", fg="#ffffff")
help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "ASL Fingerspelling Recognition App\nDeveloped with OpenCV, MediaPipe, and Tkinter"))
menubar.add_cascade(label="Help", menu=help_menu)
root.config(menu=menubar)

# ---------------------------
# Loading Splash Screen (unchanged)
# ---------------------------
splash = tk.Toplevel()
splash.overrideredirect(True)
splash.geometry("400x300+600+300")
splash.configure(bg="#1e1e1e")
splash_label = tk.Label(splash, text="Loading, please wait...", font=("Segoe UI", 24), bg="#1e1e1e", fg="#ffffff")
splash_label.pack(expand=True, fill="both")

root.withdraw()  # Hide main UI during loading

def close_splash():
    splash.destroy()
    root.deiconify()

root.after(2000, close_splash)  # Show main UI after 2 seconds

# ---------------------------
# Left Panel: Performance Metrics, Resource Info, and Log
# ---------------------------
fps_frame = tk.Frame(root, width=500, height=900, bg="#2c2f33", bd=2, relief="ridge")
fps_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
fps_frame.grid_propagate(False)

fps_frame_title = tk.Label(fps_frame, text="Performance Metrics", font=("Segoe UI", 18, "bold"),
                           bg="#2c2f33", fg="#00ffff")
fps_frame_title.pack(pady=10)

# FPS info labels
max_fps_var = tk.StringVar(value="Max FPS: 0")
min_fps_var = tk.StringVar(value="Min FPS: 0")
avg_fps_var = tk.StringVar(value="Avg FPS: 0")
pred_rate_var = tk.StringVar(value="PPF: 0")
pps_var = tk.StringVar(value="PPS: 0")

model_fps_label = tk.Label(fps_frame, text="Model: " + selected_model, font=("Segoe UI", 14),
                           bg="#2c2f33", fg="#00ff00")
model_fps_label.pack(pady=5)

max_fps_label = tk.Label(fps_frame, textvariable=max_fps_var, font=("Segoe UI", 14),
                         bg="#2c2f33", fg="#ffffff")
max_fps_label.pack(pady=5)

min_fps_label = tk.Label(fps_frame, textvariable=min_fps_var, font=("Segoe UI", 14),
                         bg="#2c2f33", fg="#ffffff")
min_fps_label.pack(pady=5)

avg_fps_label = tk.Label(fps_frame, textvariable=avg_fps_var, font=("Segoe UI", 14),
                         bg="#2c2f33", fg="#ffffff")
avg_fps_label.pack(pady=5)

pred_rate_label = tk.Label(fps_frame, textvariable=pred_rate_var, font=("Segoe UI", 14),
                           bg="#2c2f33", fg="#ffffff")
pred_rate_label.pack(pady=5)

pps_label = tk.Label(fps_frame, textvariable=pps_var, font=("Segoe UI", 14),
                     bg="#2c2f33", fg="#ffffff")
pps_label.pack(pady=5)

# Resource Info Section
resource_frame = tk.Frame(fps_frame, bg="#2c2f33")
resource_frame.pack(pady=20)

resource_title = tk.Label(resource_frame, text="Resource Info", font=("Segoe UI", 16, "bold"),
                          bg="#2c2f33", fg="#ffcc00")
resource_title.pack(pady=5)

cpu_info_var = tk.StringVar(value="CPU: Loading...")
gpu_info_var = tk.StringVar(value="GPU: Loading...")
memory_info_var = tk.StringVar(value="Memory: Loading...")

cpu_info_label = tk.Label(resource_frame, textvariable=cpu_info_var, font=("Segoe UI", 12),
                          bg="#2c2f33", fg="#ffffff")
cpu_info_label.pack(pady=2)

gpu_info_label = tk.Label(resource_frame, textvariable=gpu_info_var, font=("Segoe UI", 12),
                          bg="#2c2f33", fg="#ffffff")
gpu_info_label.pack(pady=2)

memory_info_label = tk.Label(resource_frame, textvariable=memory_info_var, font=("Segoe UI", 12),
                             bg="#2c2f33", fg="#ffffff")
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

# Log box
log_text = scrolledtext.ScrolledText(fps_frame, width=40, height=10,
                                       bg="#23272a", fg="#ffffff", font=("Segoe UI", 12))
log_text.pack(pady=10, fill="both", expand=True)
log_text.configure(state='disabled')

# ---------------------------
# Center Panel: Main Camera Feed
# ---------------------------
left_frame = tk.Frame(root, width=900, height=900, bg="#141414", bd=2, relief="ridge")
left_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
left_frame.grid_propagate(False)

main_image_label = tk.Label(left_frame, bg="#141414")
main_image_label.place(relwidth=1, relheight=1)

# ---------------------------
# Right Panel: Controls, Typed Text, and Settings
# ---------------------------
# Define these variables globally so they can be used in the Settings pop-up.
selected_model_var = tk.StringVar(value=selected_model)
auto_type_enabled_var = tk.BooleanVar(value=False)
segmentation_enabled_var = tk.BooleanVar(value=False)
ensemble_enabled_var = tk.BooleanVar(value=False)

# Global update function (for OptionMenu when ensemble mode is off)
def update_classifier_global():
    global classifier
    if not ensemble_enabled_var.get():
        sel = selected_model_var.get()
        classifier = Classifier(model_options[sel])
        model_fps_label.config(text="Model: " + sel)
        log_text.configure(state='normal')
        log_text.insert(tk.END, f"Model changed to: {sel}\n")
        log_text.see(tk.END)
        log_text.configure(state='disabled')

def change_model(*args):
    update_classifier_global()

right_frame = tk.Frame(root, width=500, height=970, bg="#1f1e1f", bd=2, relief="ridge")
right_frame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
right_frame.grid_propagate(False)
right_frame.grid_columnconfigure(0, weight=1)

# Header (Centered)
header_label = tk.Label(right_frame, text="ASL Fingerspelling Recognition", font=("Segoe UI", 20, "bold"),
                        bg="#1e1e1f", fg="#00ffff", anchor="center")
header_label.grid(row=0, column=0, pady=(15, 10), sticky="ew")

# Top Buttons Frame: Settings and Help
def open_settings():
    settings_window = tk.Toplevel(root)
    settings_window.title("Settings")
    # Elongate the settings window
    settings_window.geometry("400x500")
    settings_window.configure(bg="#1e1e1f")
    
    # Use a content frame for better layout
    content_frame = tk.Frame(settings_window, bg="#1e1e1f")
    content_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    tk.Label(content_frame, text="Model Selection:", font=("Segoe UI", 14),
             bg="#1e1e1f", fg="#ffffff").pack(pady=5, anchor="w")
    model_option_menu_settings = tk.OptionMenu(content_frame, selected_model_var, *model_options.keys(),
                                               command=lambda x: change_model())
    model_option_menu_settings.config(font=("Segoe UI", 14), bg="#333333", fg="#ffffff", highlightthickness=0)
    model_option_menu_settings["menu"].config(bg="#333333", fg="#ffffff")
    model_option_menu_settings.pack(pady=5, fill="x")
    
    segmentation_check_settings = tk.Checkbutton(content_frame, text="Enable Segmentation Filter",
                                                   variable=segmentation_enabled_var,
                                                   font=("Segoe UI", 14), bg="#1e1e1f", fg="#ffffff",
                                                   activebackground="#1e1e1f", activeforeground="#00ffff", selectcolor="#000000")
    segmentation_check_settings.pack(pady=5, anchor="w")
    
    auto_type_check = tk.Checkbutton(content_frame, text="Enable Auto Type", variable=auto_type_enabled_var,
                                     font=("Segoe UI", 14), bg="#1e1e1f", fg="#ffffff",
                                     activebackground="#1e1e1f", activeforeground="#00ffff", selectcolor="#000000")
    auto_type_check.pack(pady=5, anchor="w")
    
    # Ensemble Section
    ensemble_check_settings = tk.Checkbutton(content_frame, text="Enable Ensemble",
                                               variable=ensemble_enabled_var,
                                               font=("Segoe UI", 14), bg="#1e1e1f", fg="#ffffff",
                                               activebackground="#1e1e1f", activeforeground="#00ffff",
                                               selectcolor="#000000")
    ensemble_check_settings.pack(pady=5, anchor="w")
    
    # Add Pronunciation Accent selection
    tk.Label(content_frame, text="Pronunciation Accent:", font=("Segoe UI", 14),
             bg="#1e1e1f", fg="#ffffff").pack(pady=5, anchor="w")
    accent_option_menu = tk.OptionMenu(content_frame, selected_voice_var, *voice_dict.keys())
    accent_option_menu.config(font=("Segoe UI", 14), bg="#333333", fg="#ffffff", highlightthickness=0)
    accent_option_menu["menu"].config(bg="#333333", fg="#ffffff")
    accent_option_menu.pack(pady=5, fill="x")
    
    # Create ensemble listbox frame (initially hidden)
    ensemble_listbox_frame = tk.Frame(content_frame, bg="#1e1e1f", bd=1, relief="sunken")
    ensemble_listbox = tk.Listbox(ensemble_listbox_frame, selectmode=tk.MULTIPLE,
                                  font=("Segoe UI", 14), bg="#333333", fg="#ffffff", height=5)
    for model_name in model_options.keys():
        ensemble_listbox.insert(tk.END, model_name)
    ensemble_listbox.pack(fill="both", expand=True)
    # Initially hide the ensemble listbox frame
    ensemble_listbox_frame.pack(pady=5, fill="x")
    ensemble_listbox_frame.pack_forget()
    
    def toggle_ensemble_selection():
        print("Ensemble checkbox changed. New value:", ensemble_enabled_var.get())
        if ensemble_enabled_var.get():
            ensemble_listbox_frame.pack(pady=5, fill="x")
        else:
            ensemble_listbox_frame.pack_forget()
    
    ensemble_check_settings.config(command=toggle_ensemble_selection)
    
    # Button frame at the bottom
    button_frame = tk.Frame(content_frame, bg="#1e1e1f")
    button_frame.pack(pady=10, fill="x", side="bottom")
    
    def update_classifier():
        global classifier
        if ensemble_enabled_var.get():
            selected_indices = ensemble_listbox.curselection()
            if not selected_indices:
                selected_models = [selected_model_var.get()]
            else:
                selected_models = [ensemble_listbox.get(i) for i in selected_indices]
            model_paths = [model_options[m] for m in selected_models]
            classifier = EnsembleClassifier(model_paths)
            model_fps_label.config(text="Ensemble: " + ", ".join(selected_models))
            log_text.configure(state='normal')
            log_text.insert(tk.END, f"Ensemble classifier changed to: {', '.join(selected_models)}\n")
            log_text.see(tk.END)
            log_text.configure(state='disabled')
        else:
            sel = selected_model_var.get()
            classifier = Classifier(model_options[sel])
            model_fps_label.config(text="Model: " + sel)
            log_text.configure(state='normal')
            log_text.insert(tk.END, f"Model changed to: {sel}\n")
            log_text.see(tk.END)
            log_text.configure(state='disabled')
    
    apply_button = tk.Button(button_frame, text="Apply", command=update_classifier, font=("Segoe UI", 14),
                             bg="#333333", fg="#00ffff")
    apply_button.pack(side="left", padx=5, expand=True, fill="x")
    close_button = tk.Button(button_frame, text="Close", command=settings_window.destroy, font=("Segoe UI", 14),
                             bg="#333333", fg="#00ffff")
    close_button.pack(side="left", padx=5, expand=True, fill="x")

def open_help():
    # Create a custom Toplevel help window
    help_win = tk.Toplevel(root)
    help_win.title("Instructions")
    help_win.geometry("600x750")
    help_win.configure(bg="#1e1e1e")
    
    # Title label at the top
    title_label = tk.Label(help_win, text="Instructions", font=("Segoe UI", 18, "bold"),
                            bg="#1e1e1e", fg="#00ffff")
    title_label.pack(pady=(20, 10))
    
    # Create a content frame for padding and layout
    content_frame = tk.Frame(help_win, bg="#1e1e1e")
    content_frame.pack(padx=20, pady=10, fill="both", expand=True)
    
    # Updated instruction text including new key bindings
    instructions = (
        "1. Ensure the camera is at least 720p.\n\n"
        "2. Ensure the environment is properly lit and has a smooth background.\n\n"
        "3. Press the Space Bar key to enter the predicted letter/whitespace (when not editing the text).\n\n"
        "4. Press the Backspace key to delete the last character (when not editing the text).\n\n"
        "5. Additional Key Bindings (when editing the typed text):\n"
        "   - Enter: Pronounce the current word.\n"
        "   - Esc: Clear the current word.\n"
        "   - Delete: Delete the character to the right of the cursor.\n\n"
        "6. Settings:\n"
        "   6.1 Choose a relevant model \n (default is MobileNetV2; changing is not recommended).\n"
        "   6.2 Select the segmentation rule if in a bright room.\n"
        "   6.3 Select Auto-Type to automatically enter letters\n (3 seconds per letter).\n"
        "   6.4 Use Ensemble mode to combine predictions from multiple models\n (Note: This increases resource usage!).\n"
        "   6.5 Select the desired pronunciation accent from the settings.\n\n"
        "7. Save your work by closing the settings window."
    )
    
    # Use a Label widget with text wrapping and left justification
    help_label = tk.Label(content_frame, text=instructions, font=("Segoe UI", 12),
                          bg="#1e1e1e", fg="#ffffff", justify="left", wraplength=460)
    help_label.pack(fill="both", expand=True)
    
    # Add a Close button at the bottom
    close_btn = tk.Button(help_win, text="Close", font=("Segoe UI", 12),
                          bg="#333333", fg="#00ffff", command=help_win.destroy)
    close_btn.pack(pady=10)


# Frame to hold Settings and Help buttons side-by-side in the main UI
top_buttons_frame = tk.Frame(right_frame, bg="#1f1e1f")
top_buttons_frame.grid(row=1, column=0, pady=10, sticky="ew")
top_buttons_frame.grid_columnconfigure(0, weight=1)
top_buttons_frame.grid_columnconfigure(1, weight=1)

settings_button = tk.Button(top_buttons_frame, text="Settings", font=("Segoe UI", 14),
                             bg="#333333", fg="#00ffff", command=open_settings)
settings_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

help_button = tk.Button(top_buttons_frame, text="Help", font=("Segoe UI", 14),
                        bg="#333333", fg="#00ffff", command=open_help)
help_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

# Media Display Section (Binary and Landmarks)
FIXED_WIDTH = 280
FIXED_HEIGHT = 280

media_frame = tk.Frame(right_frame, bg="#1e1e1f")
media_frame.grid(row=2, column=0, padx=15, pady=10)
media_frame.grid_columnconfigure(0, weight=1)
media_frame.grid_columnconfigure(1, weight=1)

# Binary Image Display
binary_frame = tk.Frame(media_frame, width=FIXED_WIDTH, height=FIXED_HEIGHT, bg="#333333", bd=1, relief="sunken")
binary_frame.grid(row=0, column=0, padx=15, pady=10)
binary_frame.grid_propagate(False)
binary_title = tk.Label(binary_frame, text="Binary Image", font=("Segoe UI", 12),
                          bg="#333333", fg="#ffffff")
binary_title.pack(side="top", fill="x")
binary_label = tk.Label(binary_frame, bg="#333333")
binary_label.pack(expand=True, fill="both")

# Hand Landmarks Display
landmarks_frame = tk.Frame(media_frame, width=FIXED_WIDTH, height=FIXED_HEIGHT, bg="#333333", bd=1, relief="sunken")
landmarks_frame.grid(row=0, column=1, padx=15, pady=10)
landmarks_frame.grid_propagate(False)
landmarks_title = tk.Label(landmarks_frame, text="Landmarks", font=("Segoe UI", 12),
                            bg="#333333", fg="#ffffff")
landmarks_title.pack(side="top", fill="x")
landmarks_label = tk.Label(landmarks_frame, bg="#333333")
landmarks_label.pack(expand=True, fill="both")

# 3. Prediction Info
prediction_info_frame = tk.Frame(right_frame, bg="#1e1e1f")
prediction_info_frame.grid(row=3, column=0, padx=15, pady=10, sticky="ew")
prediction_info_frame.grid_columnconfigure(0, weight=1)

prediction_label = tk.Label(prediction_info_frame, text="", font=("Segoe UI", 16),
                            bg="#1e1e1f", fg="#ff00ff", anchor="w")
prediction_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
prediction_var = tk.StringVar(value="")
prediction_label.config(textvariable=prediction_var)

# 4. Buttons (Clear and Pronounce)
buttons_frame = tk.Frame(right_frame, bg="#1e1e1f")
buttons_frame.grid(row=4, column=0, padx=15, pady=10, sticky="ew")
buttons_frame.grid_columnconfigure(0, weight=1)
buttons_frame.grid_columnconfigure(1, weight=1)

def clear_word():
    global current_word
    current_word = ""
    log_text.configure(state='normal')
    log_text.insert(tk.END, "Clear pressed: Word cleared\n")
    log_text.see(tk.END)
    log_text.configure(state='disabled')
    # Update text box only if not editing
    if root.focus_get() != typed_text:
        typed_text.configure(state='normal')
        typed_text.delete("1.0", tk.END)
        typed_text.insert(tk.END, current_word)
        typed_text.configure(state='disabled')

clear_button = tk.Button(buttons_frame, text="Clear", font=("Segoe UI", 14),
                           bg="#333333", fg="#00ffff", command=clear_word)
clear_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

def pronounce_sentence():
    global current_word, selected_voice_var, voice_dict
    engine = pyttsx3.init()
    # Set the accent/voice based on the selected option
    if selected_voice_var.get() in voice_dict:
        engine.setProperty('voice', voice_dict[selected_voice_var.get()])
    sentence = current_word.strip()
    if sentence == "":
        log_text.configure(state='normal')
        log_text.insert(tk.END, "No sentence to pronounce.\n")
        log_text.see(tk.END)
        log_text.configure(state='disabled')
        return
    log_text.configure(state='normal')
    log_text.insert(tk.END, f"Pronouncing: {sentence}\n")
    log_text.see(tk.END)
    log_text.configure(state='disabled')
    engine.say(sentence)
    engine.runAndWait()

pronounce_button = tk.Button(buttons_frame, text="Pronounce", font=("Segoe UI", 14),
                              bg="#333333", fg="#00ffff", command=pronounce_sentence)
pronounce_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

# 5. Instructions Label
instructions_label = tk.Label(
    right_frame,
    text="1. Enter word: Space Bar  |  2. Delete last character: Backspace",
    font=("Segoe UI", 10),
    bg="#1f1f1f",
    fg="#aaaaaa"
)
instructions_label.grid(row=5, column=0, padx=15, pady=(0, 10), sticky="ew")

# 6. Typed Text Box
typed_text = scrolledtext.ScrolledText(right_frame, width=40, height=10,
                                       bg="#23272a", fg="#ffffff", font=("Segoe UI", 12))
typed_text.grid(row=6, column=0, padx=15, pady=10, sticky="ew")
# (No longer force-disabled so that a caret/pointer appears)

# --- New bindings for the text box to allow editing via cursor ---
def enable_text_editing(event):
    typed_text.configure(state='normal')

def disable_text_editing(event):
    # When focus is lost, update content from current_word and disable editing.
    typed_text.delete("1.0", tk.END)
    typed_text.insert("1.0", current_word)
    typed_text.configure(state='disabled')

typed_text.bind("<FocusIn>", enable_text_editing)
typed_text.bind("<FocusOut>", disable_text_editing)

def on_delete_key_in_text(event):
    global current_word
    index = typed_text.index("insert")  # e.g., "1.5"
    line, col = index.split('.')
    col = int(col)
    if col < len(current_word):
        current_word = current_word[:col] + current_word[col+1:]
        typed_text.delete("1.0", tk.END)
        typed_text.insert("1.0", current_word)
        typed_text.mark_set("insert", f"1.{col}")
    return "break"

def on_backspace_in_text(event):
    global current_word
    index = typed_text.index("insert")
    line, col = index.split('.')
    col = int(col)
    if col > 0:
        new_index = col - 1
        current_word = current_word[:new_index] + current_word[new_index+1:]
        typed_text.delete("1.0", tk.END)
        typed_text.insert("1.0", current_word)
        typed_text.mark_set("insert", f"1.{new_index}")
    return "break"

def on_return_in_text(event):
    pronounce_sentence()
    return "break"

def on_escape_in_text(event):
    clear_word()
    return "break"

typed_text.bind("<Delete>", on_delete_key_in_text)
typed_text.bind("<BackSpace>", on_backspace_in_text)
typed_text.bind("<Return>", on_return_in_text)
typed_text.bind("<Escape>", on_escape_in_text)

# ---------------------------
# Key Binding Functions for Global Shortcuts
# ---------------------------
def on_space_press(event):
    # If the text box is being edited, do not process global key binding.
    if root.focus_get() == typed_text:
        return
    global current_word, current_prediction, last_space_time
    current_time = time.time()
    if current_time - last_space_time < space_cooldown:
        return
    last_space_time = current_time
    if current_prediction != "":
        current_word += current_prediction
        log_text.configure(state='normal')
        log_text.insert(tk.END, f"Entered: {current_prediction}\n")
        log_text.see(tk.END)
        log_text.configure(state='disabled')
    else:
        current_word += " "
        log_text.configure(state='normal')
        log_text.insert(tk.END, "Entered: [space]\n")
        log_text.see(tk.END)
        log_text.configure(state='disabled')
    if root.focus_get() != typed_text:
        typed_text.configure(state='normal')
        typed_text.delete("1.0", tk.END)
        typed_text.insert(tk.END, current_word)
        typed_text.configure(state='disabled')

def on_backspace(event):
    # If the text box is being edited, let its own binding handle backspace.
    if root.focus_get() == typed_text:
        return
    global current_word
    if current_word:
        current_word = current_word[:-1]
        log_text.configure(state='normal')
        log_text.insert(tk.END, "Backspace pressed: Removed last character\n")
        log_text.see(tk.END)
        log_text.configure(state='disabled')
        if root.focus_get() != typed_text:
            typed_text.configure(state='normal')
            typed_text.delete("1.0", tk.END)
            typed_text.insert(tk.END, current_word)
            typed_text.configure(state='disabled')

root.bind("<space>", on_space_press)
root.bind("<BackSpace>", on_backspace)

# --- New global bindings for Enter and Escape keys ---
root.bind("<Return>", lambda event: pronounce_sentence())
root.bind("<Escape>", lambda event: clear_word())

# ---------------------------
# Main update_frame() function
# ---------------------------
def update_frame():
    global current_prediction, last_frame_time, fps_values, frame_counter, pred_counter, last_fps_report_time
    global auto_type_start_time, auto_type_last_prediction, current_word

    success, img = cap.read()
    if not success:
        root.after(10, update_frame)
        return

    # FPS Measurement
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

    # ---------------------------
    # Auto Type Functionality
    # ---------------------------
    if auto_type_enabled_var.get():
        if hands:
            if auto_type_start_time is None or current_prediction != auto_type_last_prediction:
                auto_type_start_time = current_time
                auto_type_last_prediction = current_prediction
            else:
                if current_time - auto_type_start_time >= 1.5:
                    if current_prediction != "":
                        current_word += current_prediction
                        log_text.configure(state='normal')
                        log_text.insert(tk.END, f"Auto typed: {current_prediction}\n")
                        log_text.see(tk.END)
                        log_text.configure(state='disabled')
                        if root.focus_get() != typed_text:
                            typed_text.configure(state='normal')
                            typed_text.delete("1.0", tk.END)
                            typed_text.insert(tk.END, current_word)
                            typed_text.configure(state='disabled')
                        auto_type_start_time = current_time
        else:
            auto_type_start_time = None
            auto_type_last_prediction = None
    else:
        auto_type_start_time = None
        auto_type_last_prediction = None

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
        pps = pred_counter

        max_fps_var.set(f"Max FPS: {max_fps:.2f}")
        min_fps_var.set(f"Min FPS: {min_fps:.2f}")
        avg_fps_var.set(f"Avg FPS: {avg_fps:.2f}")
        pred_rate_var.set(f"Predictions per frame: {pred_rate:.2f}")
        pps_var.set(f"PPS: {pps}")

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
# Status Bar (New)
# ---------------------------
status_var = tk.StringVar(value="Ready")
status_bar = tk.Label(root, textvariable=status_var, font=("Segoe UI", 12), bg="#2c2f33", fg="#ffffff", bd=1, relief="sunken")
status_bar.grid(row=1, column=0, columnspan=3, sticky="ew")

# ---------------------------
# Clean up on exit (unchanged)
# ---------------------------
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

update_frame()
root.mainloop()

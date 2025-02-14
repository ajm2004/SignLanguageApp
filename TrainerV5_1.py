import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import (
    MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large,
    ResNet50, EfficientNetB0, Xception, DenseNet121, VGG16, VGG19,
    NASNetMobile, InceptionV3
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tkinter import Tk, StringVar, DoubleVar, IntVar, BooleanVar, END, Text
import tkinter.ttk as ttk
from tkinter import filedialog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import json
import time
from datetime import datetime

# For evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

# ---------------------------
# Grad-CAM helper functions
# ---------------------------
def find_last_conv_layer(model):
    """Finds the name of the last Conv2D layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates a Grad-CAM heatmap for a given image array and model."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

def save_gradcam_overlay(img_path, heatmap, output_path, alpha=0.4):
    """Overlays the heatmap on the original image and saves the result."""
    import cv2
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.uint8(superimposed_img)
    plt.imsave(output_path, superimposed_img)

# ---------------------------
# Main Application Class
# ---------------------------
class ASLTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Image Trainer")
        self.root.geometry("1150x950")
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Shared stop event for training interruption
        self.stop_event = threading.Event()
        
        # Variables
        self.dataset_path = StringVar()
        self.train_test_split = StringVar(value="80/20")
        self.epochs = IntVar(value=50)
        self.batch_size = IntVar(value=16)
        self.learning_rate = DoubleVar(value=0.001)
        self.fine_tune = BooleanVar(value=False)
        self.use_cross_validation = BooleanVar(value=False)
        self.num_folds = IntVar(value=5)
        # New variable for model save directory (default provided)
        self.model_save_dir = StringVar(value=r"TrainedBinaryNewModel")
        
        # Model selection variables
        self.model_choice = StringVar(value="BasicCNN")
        self.selected_models = []  # list of model names
        self.model_items = {}  # mapping: model name -> treeview item id

        # Models dictionary (added FusionModel)
        self.models = {
            "BasicCNN": self.build_basic_cnn,
            "MobileNet": MobileNet,
            "MobileNetV2": MobileNetV2,
            "MobileNetV3Small": MobileNetV3Small,
            "MobileNetV3Large": MobileNetV3Large,
            "EfficientNetB0": EfficientNetB0,
            "ResNet50": ResNet50,
            "Xception": Xception,
            "DenseNet121": DenseNet121,
            "VGG16": VGG16,
            "VGG19": VGG19,
            "NASNetMobile": NASNetMobile,
            "InceptionV3": InceptionV3,
            "FusionModel": self.build_fusion_model
        }
        
        # Layout: Two main panels (left and right)
        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill='both', expand=True)
        
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="ns", padx=10, pady=10)
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left Panel: Selected Models (Treeview) and Log Box (scrollable)
        sel_models_label = ttk.Label(left_frame, text="Selected Models", font=("Arial", 12, "bold"))
        sel_models_label.pack(pady=5)
        
        # Treeview with columns: Model, Status, Val Accuracy, Val Loss, Total Time
        columns = ("Model", "Status", "Val Accuracy", "Val Loss", "Total Time")
        self.models_tree = ttk.Treeview(left_frame, columns=columns, show="headings", height=8)
        for col in columns:
            self.models_tree.heading(col, text=col)
        self.models_tree.column("Model", anchor="center", width=120)
        self.models_tree.column("Status", anchor="center", width=100)
        self.models_tree.column("Val Accuracy", anchor="center", width=90)
        self.models_tree.column("Val Loss", anchor="center", width=90)
        self.models_tree.column("Total Time", anchor="center", width=100)
        self.models_tree.pack(pady=5, fill="x")
        
        # Log Box
        log_label = ttk.Label(left_frame, text="Training Log", font=("Arial", 12, "bold"))
        log_label.pack(pady=5)
        log_frame = ttk.Frame(left_frame)
        log_frame.pack(fill='both', expand=True)
        self.log_text = Text(log_frame, height=15, wrap="word")
        self.log_text.pack(side='left', fill='both', expand=True)
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.pack(side='right', fill='y')
        self.log_text.config(yscrollcommand=log_scroll.set)
        
        # Right Panel: Parameter selections and control buttons
        title_label = ttk.Label(right_frame, text="ASL Image Trainer", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Dataset selection
        dataset_button = ttk.Button(right_frame, text="Select Dataset", command=self.select_dataset)
        dataset_button.pack(pady=5)
        dataset_path_label = ttk.Label(right_frame, textvariable=self.dataset_path)
        dataset_path_label.pack(pady=5)
        
        # Model Save Directory selection
        save_dir_button = ttk.Button(right_frame, text="Select Save Directory", command=self.select_model_save_dir)
        save_dir_button.pack(pady=5)
        save_dir_label = ttk.Label(right_frame, textvariable=self.model_save_dir)
        save_dir_label.pack(pady=5)
        
        # Train/Test split dropdown
        split_label = ttk.Label(right_frame, text="Train/Test Split:")
        split_label.pack(pady=5)
        split_options = ["80/20", "75/25", "70/30", "60/40"]
        split_menu = ttk.OptionMenu(right_frame, self.train_test_split, self.train_test_split.get(), *split_options)
        split_menu.pack(pady=5)
        
        # Model selection dropdown and add/remove buttons
        model_select_label = ttk.Label(right_frame, text="Select Model to Add:")
        model_select_label.pack(pady=5)
        model_menu = ttk.OptionMenu(right_frame, self.model_choice, self.model_choice.get(), *self.models.keys())
        model_menu.pack(pady=5)
        add_model_button = ttk.Button(right_frame, text="Add Model", command=self.add_model)
        add_model_button.pack(pady=5)
        remove_model_button = ttk.Button(right_frame, text="Remove Selected Model", command=self.remove_selected_model)
        remove_model_button.pack(pady=5)
        
        # Training parameters
        epochs_label = ttk.Label(right_frame, text="Epochs:")
        epochs_label.pack(pady=5)
        epochs_entry = ttk.Entry(right_frame, textvariable=self.epochs)
        epochs_entry.pack(pady=5)
        batch_label = ttk.Label(right_frame, text="Batch Size:")
        batch_label.pack(pady=5)
        batch_entry = ttk.Entry(right_frame, textvariable=self.batch_size)
        batch_entry.pack(pady=5)
        lr_label = ttk.Label(right_frame, text="Learning Rate:")
        lr_label.pack(pady=5)
        lr_entry = ttk.Entry(right_frame, textvariable=self.learning_rate)
        lr_entry.pack(pady=5)
        fine_tune_checkbox = ttk.Checkbutton(right_frame, text="Fine-tune Model", variable=self.fine_tune)
        fine_tune_checkbox.pack(pady=5)
        cv_checkbox = ttk.Checkbutton(right_frame, text="Use Cross Validation", variable=self.use_cross_validation)
        cv_checkbox.pack(pady=5)
        folds_label = ttk.Label(right_frame, text="Number of Folds:")
        folds_label.pack(pady=5)
        folds_entry = ttk.Entry(right_frame, textvariable=self.num_folds)
        folds_entry.pack(pady=5)
        
        # Control buttons: Train and Stop
        self.train_button = ttk.Button(right_frame, text="Train Models", command=self.train_model)
        self.train_button.pack(pady=10)
        self.stop_button = ttk.Button(right_frame, text="Stop Training", command=self.stop_training, state="disabled")
        self.stop_button.pack(pady=5)
        
        # Progress bar and time labels
        self.progress_bar = ttk.Progressbar(right_frame, orient='horizontal', mode='determinate', length=300)
        self.progress_bar.pack(pady=10)
        self.time_label = ttk.Label(right_frame, text="Estimated Time Remaining: N/A")
        self.time_label.pack(pady=5)
        # New label for time elapsed
        self.elapsed_label = ttk.Label(right_frame, text="Time Elapsed: 0 sec")
        self.elapsed_label.pack(pady=5)
        
        self.status_label = ttk.Label(right_frame, text="Status: Waiting for input...", wraplength=400, justify="left")
        self.status_label.pack(pady=10)
        self.progress_label = ttk.Label(right_frame, text="", wraplength=400, justify="left")
        self.progress_label.pack(pady=10)
    
    def select_dataset(self):
        path = filedialog.askdirectory()
        if path:
            self.dataset_path.set(path)
    
    def select_model_save_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.model_save_dir.set(path)
    
    def add_model(self):
        """Add a model to the sequence (max 7)."""
        selected = self.model_choice.get()
        if len(self.selected_models) >= 7:
            self.log("Maximum of 7 models already selected.")
        elif selected in self.selected_models:
            self.log(f"Model '{selected}' is already in the sequence.")
        else:
            self.selected_models.append(selected)
            # Insert into treeview with initial values
            item_id = self.models_tree.insert("", END, values=(selected, "Waiting", "-", "-", "-"))
            self.model_items[selected] = item_id
            self.log(f"Added model '{selected}' to the sequence.")
    
    def remove_selected_model(self):
        """Remove the selected model from the treeview."""
        selected_item = self.models_tree.selection()
        if not selected_item:
            self.log("No model selected to remove.")
            return
        model_name = self.models_tree.item(selected_item, "values")[0]
        self.models_tree.delete(selected_item)
        if model_name in self.selected_models:
            self.selected_models.remove(model_name)
            self.model_items.pop(model_name, None)
        self.log(f"Removed model '{model_name}' from the sequence.")
    
    def log(self, message):
        """Append a message to the log text box."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"
        self.log_text.insert(END, full_message)
        self.log_text.see(END)
    
    def build_basic_cnn(self, input_shape, num_classes):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            Dense(128),
            LeakyReLU(alpha=0.1),
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ])
        return model
    
    def build_fusion_model(self, input_shape, num_classes, fine_tune):
        input_layer = tf.keras.layers.Input(shape=input_shape)
        base_model1 = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_layer)
        if not fine_tune:
            for layer in base_model1.layers:
                layer.trainable = False
        x1 = GlobalAveragePooling2D()(base_model1.output)
        base_model2 = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)
        if not fine_tune:
            for layer in base_model2.layers:
                layer.trainable = False
        x2 = GlobalAveragePooling2D()(base_model2.output)
        from tensorflow.keras.layers import Concatenate
        merged = Concatenate()([x1, x2])
        x = Dense(256, activation='relu')(merged)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=outputs)
        return model
    
    def build_model(self, model_name, input_shape, num_classes, fine_tune):
        if model_name == "BasicCNN":
            return self.build_basic_cnn(input_shape, num_classes)
        elif model_name == "FusionModel":
            return self.build_fusion_model(input_shape, num_classes, fine_tune)
        else:
            base_model = self.models[model_name](weights='imagenet', include_top=False, input_shape=input_shape)
            if not fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            x = GlobalAveragePooling2D()(base_model.output)
            x = Dense(128, activation="relu")(x)
            predictions = Dense(num_classes, activation="softmax")(x)
            return Model(inputs=base_model.input, outputs=predictions)
    
    def train_model(self):
        if not self.dataset_path.get():
            self.status_label.config(text="Please select a dataset folder.")
            return
        if len(self.selected_models) == 0:
            self.status_label.config(text="Please add at least one model to the sequence.")
            return
        
        self.stop_event.clear()
        self.train_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        training_thread = threading.Thread(target=self._train_model_thread)
        training_thread.start()
    
    def stop_training(self):
        self.stop_event.set()
        self.log("Stop requested by user.")
        self.status_label.config(text="Stop requested. Halting training...")
        self.stop_button.config(state="disabled")
    
    def _train_model_thread(self):
        epochs = self.epochs.get()
        batch_size = self.batch_size.get()
        learning_rate = self.learning_rate.get()
        fine_tune = self.fine_tune.get()
        split_options = {"80/20": 0.2, "75/25": 0.25, "70/30": 0.3, "60/40": 0.4}
        validation_split = split_options.get(self.train_test_split.get(), 0.2)
        
        models_to_train = self.selected_models[:]
        for model_name in models_to_train:
            if self.stop_event.is_set():
                self.update_model_status(model_name, "Stopped")
                self.log(f"Training stopped for model {model_name}.")
                continue
            
            self.update_model_status(model_name, "Preparing")
            self.status_label.config(text=f"Preparing data for model {model_name}...")
            self.log(f"Starting training for model: {model_name}")
            
            if model_name == "BasicCNN":
                img_size = (64, 64)
            elif model_name in ["Xception", "InceptionV3"]:
                img_size = (299, 299)
            else:
                img_size = (224, 224)
            
            if self.use_cross_validation.get():
                self.perform_cross_validation(model_name, epochs, batch_size, learning_rate, fine_tune)
                self.log(f"Cross validation training complete for {model_name}.")
                continue
            
            self.log(f"Preparing ImageDataGenerator with validation_split={validation_split} for {model_name}...")
            datagen = ImageDataGenerator(
                rescale=1.0/255,
                validation_split=validation_split,
                rotation_range=5,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.2,
                horizontal_flip=False
            )
            train_generator = datagen.flow_from_directory(
                self.dataset_path.get(),
                target_size=img_size,
                batch_size=batch_size,
                class_mode='categorical',
                subset='training'
            )
            validation_generator = datagen.flow_from_directory(
                self.dataset_path.get(),
                target_size=img_size,
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False
            )
            num_classes = len(train_generator.class_indices)
            model = self.build_model(model_name, (*img_size, 3), num_classes, fine_tune)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='categorical_crossentropy',
                          metrics=['accuracy', 
                                   tf.keras.metrics.Precision(name='precision'),
                                   tf.keras.metrics.Recall(name='recall')])
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
            self.update_model_status(model_name, "Training")
            self.status_label.config(text=f"Training model {model_name}...")
            start_time = time.time()
            history = model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=epochs,
                verbose=1,
                callbacks=[self.TrainingCallback(self.progress_label, self.progress_bar, 
                                                   self.time_label, self.elapsed_label, epochs, self.stop_event),
                           early_stopping]
            )
            elapsed = time.time() - start_time
            self.log(f"Training for {model_name} completed in {elapsed:.1f} seconds.")
            
            if self.stop_event.is_set():
                self.update_model_status(model_name, "Stopped")
                continue
            
            self.update_model_status(model_name, "Saving")
            self.status_label.config(text=f"Training complete for {model_name}. Saving model and training details...")
            model_dir = self.model_save_dir.get()  # use selected save directory
            os.makedirs(model_dir, exist_ok=True)
            self.log(f"Saving model {model_name} to folder: {model_dir}")
            model_file_name = f"{model_name}_model.h5"
            model_save_path = os.path.join(model_dir, model_file_name)
            try:
                model.save(model_save_path)
                self.log(f"Model {model_name} saved successfully.")
            except Exception as e:
                self.log(f"Error saving model {model_name}: {e}")
                self.status_label.config(text=f"Error saving model: {e}")
                continue
            
            training_details = {
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "fine_tune": fine_tune,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "train_val_split": f"Training ({int((1-validation_split)*100)}%) | Validation ({int(validation_split*100)}%)",
                "training_history": history.history
            }
            details_file_name = f"{model_name}_training_details.json"
            details_save_path = os.path.join(model_dir, details_file_name)
            try:
                with open(details_save_path, "w") as details_file:
                    json.dump(training_details, details_file, indent=4)
                self.log(f"Training details saved for {model_name}.")
            except Exception as e:
                self.log(f"Error saving training details for {model_name}: {e}")
                self.status_label.config(text=f"Error saving training details: {e}")
                continue
            
            results_folder = os.path.join(os.getcwd(), f"{model_name}_results")
            os.makedirs(results_folder, exist_ok=True)
            self.plot_training_history(history, results_folder, model_name)
            self.evaluate_and_save_metrics(model, validation_generator, model_name, self.dataset_path.get(), results_folder)
            
            try:
                class_folders = sorted([d for d in os.listdir(self.dataset_path.get()) if os.path.isdir(os.path.join(self.dataset_path.get(), d))])
                if len(class_folders) > 0:
                    sample_class_dir = os.path.join(self.dataset_path.get(), class_folders[0])
                    sample_img = None
                    for file in os.listdir(sample_class_dir):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            sample_img = os.path.join(sample_class_dir, file)
                            break
                    if sample_img is not None:
                        self.generate_and_save_gradcam_heatmap(model, sample_img, img_size, model_name, results_folder)
                        self.log(f"Grad-CAM heatmap generated for {model_name}.")
                    else:
                        self.log("No image found in the sample class folder for Grad-CAM.")
                else:
                    self.log("No class folders found in the dataset directory.")
            except Exception as e:
                self.log(f"Error generating Grad-CAM heatmap for {model_name}: {e}")
            
            best_val_acc = max(history.history.get('val_accuracy', [0]))
            best_val_loss = min(history.history.get('val_loss', [0]))
            self.update_model_metrics(model_name, best_val_acc, best_val_loss)
            self.update_model_time(model_name, elapsed)
            self.update_model_status(model_name, "Completed")
            self.log(f"Training complete for model: {model_name}\n{'-'*50}")
        
        self.status_label.config(text="All models training complete.")
        self.train_button.config(state="normal")
        self.stop_button.config(state="disabled")
    
    def perform_cross_validation(self, model_name, epochs, batch_size, learning_rate, fine_tune):
        self.status_label.config(text="Performing cross validation...")
        file_paths, labels, class_names = self.load_image_paths_and_labels(self.dataset_path.get())
        if model_name == "BasicCNN":
            img_size = (64, 64)
        elif model_name in ["Xception", "InceptionV3"]:
            img_size = (299, 299)
        else:
            img_size = (224, 224)
        
        def load_and_preprocess_image(path, label):
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3)
            image.set_shape([None, None, 3])
            image = tf.image.resize(image, img_size)
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
        
        kf = KFold(n_splits=self.num_folds.get(), shuffle=True, random_state=42)
        all_y_true = []
        all_y_pred = []
        fold_metrics = []
        fold_idx = 1
        
        for train_index, val_index in kf.split(file_paths):
            self.status_label.config(text=f"Training fold {fold_idx}/{self.num_folds.get()} for {model_name}...")
            train_paths = file_paths[train_index]
            train_labels = labels[train_index]
            val_paths = file_paths[val_index]
            val_labels = labels[val_index]
            
            train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
            train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
            train_ds = train_ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
            val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            num_classes = len(class_names)
            model = self.build_model(model_name, (*img_size, 3), num_classes, fine_tune)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=1)
            y_pred_prob = model.predict(val_ds)
            y_pred = np.argmax(y_pred_prob, axis=1)
            all_y_true.extend(val_labels)
            all_y_pred.extend(y_pred)
            
            fold_accuracy = accuracy_score(val_labels, y_pred)
            fold_precision = precision_score(val_labels, y_pred, average='weighted')
            fold_recall = recall_score(val_labels, y_pred, average='weighted')
            fold_f1 = f1_score(val_labels, y_pred, average='weighted')
            fold_metrics.append((fold_accuracy, fold_precision, fold_recall, fold_f1))
            fold_idx += 1
        
        overall_accuracy = accuracy_score(all_y_true, all_y_pred)
        overall_precision = precision_score(all_y_true, all_y_pred, average='weighted')
        overall_recall = recall_score(all_y_true, all_y_pred, average='weighted')
        overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
        
        report = classification_report(all_y_true, all_y_pred, target_names=class_names)
        cm = confusion_matrix(all_y_true, all_y_pred)
        
        cv_results_folder = os.path.join(os.getcwd(), f"{model_name}_cv_results")
        os.makedirs(cv_results_folder, exist_ok=True)
        txt_file = os.path.join(cv_results_folder, f"{model_name}_cv_evaluation.txt")
        with open(txt_file, "w") as f:
            f.write(f"Cross Validation Results for model: {model_name}\n")
            f.write(f"Number of Folds: {self.num_folds.get()}\n")
            f.write("Fold Metrics (Accuracy, Precision, Recall, F1):\n")
            for i, m in enumerate(fold_metrics, start=1):
                f.write(f"Fold {i}: {m}\n")
            f.write("\nOverall Metrics:\n")
            f.write(f"Accuracy: {overall_accuracy:.4f}\n")
            f.write(f"Precision: {overall_precision:.4f}\n")
            f.write(f"Recall: {overall_recall:.4f}\n")
            f.write(f"F1 Score: {overall_f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        
        plt.figure(figsize=(8,6))
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metrics_values = [overall_accuracy, overall_precision, overall_recall, overall_f1]
        sns.barplot(x=metrics_names, y=metrics_values)
        plt.ylim(0, 1)
        plt.title('Cross Validation Overall Metrics')
        metrics_graph_path = os.path.join(cv_results_folder, f"{model_name}_cv_metrics_bar.png")
        plt.savefig(metrics_graph_path)
        plt.close()
        
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Cross Validation Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        cm_path = os.path.join(cv_results_folder, f"{model_name}_cv_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        
        self.status_label.config(text=f"Cross validation complete for {model_name}. Results saved.")
    
    def generate_and_save_gradcam_heatmap(self, model, sample_img_path, img_size, model_name, results_folder):
        from tensorflow.keras.preprocessing import image
        img = image.load_img(sample_img_path, target_size=img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        last_conv_layer_name = find_last_conv_layer(model)
        if last_conv_layer_name is None:
            self.log("No convolutional layer found in the model; Grad-CAM cannot be applied.")
            return

        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        output_path = os.path.join(results_folder, f"{model_name}_gradcam_heatmap.png")
        save_gradcam_overlay(sample_img_path, heatmap, output_path)
        self.log(f"Grad-CAM heatmap saved to: {output_path}")
    
    def plot_training_history(self, history, result_folder, model_name):
        epochs_range = range(len(history.history['accuracy']))
        if 'precision' in history.history and 'recall' in history.history:
            train_f1 = [2*(p*r)/(p+r) if (p+r) > 0 else 0 
                        for p, r in zip(history.history['precision'], history.history['recall'])]
            val_f1 = [2*(p*r)/(p+r) if (p+r) > 0 else 0 
                      for p, r in zip(history.history['val_precision'], history.history['val_recall'])]
        else:
            train_f1 = [0] * len(epochs_range)
            val_f1 = [0] * len(epochs_range)
        
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        axs[0, 0].plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
        axs[0, 0].plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
        axs[0, 0].legend(loc='lower right')
        axs[0, 0].set_title('Accuracy')
        
        axs[0, 1].plot(epochs_range, history.history['loss'], label='Training Loss')
        axs[0, 1].plot(epochs_range, history.history['val_loss'], label='Validation Loss')
        axs[0, 1].legend(loc='upper right')
        axs[0, 1].set_title('Loss')
        
        if 'precision' in history.history:
            axs[0, 2].plot(epochs_range, history.history['precision'], label='Training Precision')
            axs[0, 2].plot(epochs_range, history.history['val_precision'], label='Validation Precision')
            axs[0, 2].legend(loc='lower right')
            axs[0, 2].set_title('Precision')
        else:
            axs[0, 2].set_visible(False)
        
        if 'recall' in history.history:
            axs[1, 0].plot(epochs_range, history.history['recall'], label='Training Recall')
            axs[1, 0].plot(epochs_range, history.history['val_recall'], label='Validation Recall')
            axs[1, 0].legend(loc='lower right')
            axs[1, 0].set_title('Recall')
        else:
            axs[1, 0].set_visible(False)
        
        if 'precision' in history.history and 'recall' in history.history:
            axs[1, 1].plot(epochs_range, train_f1, label='Training F1')
            axs[1, 1].plot(epochs_range, val_f1, label='Validation F1')
            axs[1, 1].legend(loc='lower right')
            axs[1, 1].set_title('F1 Score')
        else:
            axs[1, 1].set_visible(False)
        
        axs[1, 2].axis('off')
        
        plt.suptitle('Training Metrics Over Epochs')
        training_plot_path = os.path.join(result_folder, f"{model_name}_training_metrics.png")
        plt.savefig(training_plot_path)
        plt.close()
    
    def evaluate_and_save_metrics(self, model, validation_generator, model_name, dataset_path, result_folder):
        validation_generator.reset()
        Y_pred = model.predict(validation_generator, verbose=1)
        y_pred = np.argmax(Y_pred, axis=1)
        y_true = validation_generator.classes
        class_names = list(validation_generator.class_indices.keys())
        
        report = classification_report(y_true, y_pred, target_names=class_names)
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        txt_file = os.path.join(result_folder, f"{model_name}_evaluation.txt")
        with open(txt_file, "w") as f:
            f.write(f"Model Name: {model_name}\n")
            f.write(f"Dataset Path: {dataset_path}\n")
            f.write("Train-Test Split: Training (80%) | Validation (20%)\n\n")
            f.write("Evaluation Metrics:\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        
        plt.figure(figsize=(8,6))
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metrics_values = [accuracy, precision, recall, f1]
        sns.barplot(x=metrics_names, y=metrics_values)
        plt.ylim(0, 1)
        plt.title('Evaluation Metrics')
        metrics_graph_path = os.path.join(result_folder, f"{model_name}_metrics_bar.png")
        plt.savefig(metrics_graph_path)
        plt.close()

        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        cm_path = os.path.join(result_folder, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
    
    def load_image_paths_and_labels(self, dataset_dir):
        file_paths = []
        labels = []
        class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
        for label_idx, cls in enumerate(class_names):
            cls_dir = os.path.join(dataset_dir, cls)
            for file in os.listdir(cls_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_paths.append(os.path.join(cls_dir, file))
                    labels.append(label_idx)
        return np.array(file_paths), np.array(labels), class_names
    
    def update_model_status(self, model_name, status):
        if model_name in self.model_items:
            item_id = self.model_items[model_name]
            current = list(self.models_tree.item(item_id, "values"))
            current[1] = status
            self.models_tree.item(item_id, values=current)
    
    def update_model_metrics(self, model_name, val_acc, val_loss):
        if model_name in self.model_items:
            item_id = self.model_items[model_name]
            current = list(self.models_tree.item(item_id, "values"))
            current[2] = f"{val_acc:.4f}"
            current[3] = f"{val_loss:.4f}"
            self.models_tree.item(item_id, values=current)
    
    def update_model_time(self, model_name, elapsed):
        if model_name in self.model_items:
            item_id = self.model_items[model_name]
            current = list(self.models_tree.item(item_id, "values"))
            current[4] = f"{elapsed:.1f} sec"
            self.models_tree.item(item_id, values=current)
    
    class TrainingCallback(tf.keras.callbacks.Callback):
        def __init__(self, progress_label, progress_bar, time_label, elapsed_label, total_epochs, stop_event):
            super().__init__()
            self.progress_label = progress_label
            self.progress_bar = progress_bar
            self.time_label = time_label   # For estimated time remaining
            self.elapsed_label = elapsed_label  # For time elapsed
            self.total_epochs = total_epochs
            self.stop_event = stop_event
            self.epoch_times = []
        
        def on_train_begin(self, logs=None):
            self.training_start_time = time.time()
        
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start = time.time()
        
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            epoch_time = time.time() - self.epoch_start
            self.epoch_times.append(epoch_time)
            avg_epoch = np.mean(self.epoch_times)
            remaining_epochs = self.total_epochs - (epoch + 1)
            est_remaining = remaining_epochs * avg_epoch
            progress_val = ((epoch + 1) / self.total_epochs) * 100
            self.progress_bar.after(0, lambda: self.progress_bar.config(value=progress_val))
            self.time_label.after(0, lambda: self.time_label.config(text=f"Estimated Time Remaining: {int(est_remaining)} sec"))
            elapsed = time.time() - self.training_start_time
            self.elapsed_label.after(0, lambda: self.elapsed_label.config(text=f"Time Elapsed: {int(elapsed)} sec"))
            text = (f"Epoch {epoch + 1}/{self.total_epochs}\n"
                    f"Training Acc: {logs.get('accuracy', 0):.4f}, Loss: {logs.get('loss', 0):.4f}\n"
                    f"Validation Acc: {logs.get('val_accuracy', 0):.4f}, Loss: {logs.get('val_loss', 0):.4f}")
            self.progress_label.after(0, lambda: self.progress_label.config(text=text))
            if self.stop_event.is_set():
                self.model.stop_training = True

if __name__ == "__main__":
    print("Available GPU devices:", tf.config.list_physical_devices('GPU'))
    root = Tk()
    app = ASLTrainerApp(root)
    root.mainloop()

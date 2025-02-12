import os
import time
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
from datetime import datetime
import json
import threading

# For evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

# For plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# For UI
import tkinter as tk
from tkinter import Tk, Label, Button, filedialog, StringVar, DoubleVar, IntVar, OptionMenu, Entry, Checkbutton, BooleanVar
from tkinter import messagebox
from tkinter import scrolledtext
from tkinter import END
from tkinter import N, S, E, W
from tkinter import Listbox
from tkinter import MULTIPLE
from tkinter import VERTICAL
from tkinter import HORIZONTAL
from tkinter import Scrollbar
from tkinter import RIGHT, LEFT, Y, BOTH
from tkinter import Frame
from tkinter import TOP, BOTTOM, X
# Use themed widgets for a modern look.
from tkinter import ttk

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
    # Normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

def save_gradcam_overlay(img_path, heatmap, output_path, alpha=0.4):
    """Overlays the heatmap on the original image and saves the result."""
    import cv2
    img = cv2.imread(img_path)
    # Convert BGR to RGB
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
        self.root.geometry("800x800")
        self.root.minsize(600, 600)

        # ---------------------------
        # Variables
        # ---------------------------
        self.dataset_path = StringVar()
        self.test_dataset_path = StringVar()
        self.model_name = StringVar(value="BasicCNN")
        self.epochs = IntVar(value=50)
        self.batch_size = IntVar(value=16)
        self.learning_rate = DoubleVar(value=0.001)
        self.fine_tune = BooleanVar(value=False)
        self.use_cross_validation = BooleanVar(value=False)
        self.num_folds = IntVar(value=5)
        # New options:
        self.use_separate_test = BooleanVar(value=False)
        self.split_ratio = StringVar(value="80-20")
        self.train_sequentially = BooleanVar(value=False)

        # Mapping for split ratios
        self.split_options = ["80-20", "75-25", "70-30"]
        self.split_mapping = {"80-20": 0.2, "75-25": 0.25, "70-30": 0.3}

        # Updated models dictionary with MobileNetV3 and VGG19 added.
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
            "InceptionV3": InceptionV3
        }

        # ---------------------------
        # Layout Setup
        # ---------------------------
        # Create a main frame to hold the left (sequential queue) and right (controls) panels.
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Left panel: Sequential Training Queue
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side='left', fill='y', padx=10, pady=10)
        ttk.Label(self.left_frame, text="Sequential Training Queue", font=("Arial", 12, "bold")).pack(pady=5)
        self.queue_tree = ttk.Treeview(self.left_frame, columns=("Model", "Status"), show='headings', height=10)
        self.queue_tree.heading("Model", text="Model")
        self.queue_tree.heading("Status", text="Status")
        self.queue_tree.column("Model", width=120)
        self.queue_tree.column("Status", width=120)
        self.queue_tree.pack(pady=5)

        # Right panel: Training Options and Controls
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)

        # Title and dataset selection
        ttk.Label(self.right_frame, text="ASL Image Trainer", font=("Arial", 16, "bold")).pack(pady=10)
        ttk.Button(self.right_frame, text="Select Training Dataset", command=self.select_dataset).pack(pady=5)
        ttk.Label(self.right_frame, textvariable=self.dataset_path).pack(pady=5)

        # Option: Use separate test dataset
        ttk.Checkbutton(self.right_frame, text="Use Separate Test Dataset", variable=self.use_separate_test, command=self.toggle_test_dataset).pack(pady=5)
        self.test_dataset_button = ttk.Button(self.right_frame, text="Select Test Dataset", command=self.select_test_dataset)
        self.test_dataset_label = ttk.Label(self.right_frame, textvariable=self.test_dataset_path)
        # Initially hidden if not selected:
        if self.use_separate_test.get():
            self.test_dataset_button.pack(pady=5)
            self.test_dataset_label.pack(pady=5)

        # Train-Test Split Ratio
        ttk.Label(self.right_frame, text="Train-Test Split Ratio:").pack(pady=5)
        self.split_dropdown = ttk.OptionMenu(self.right_frame, self.split_ratio, self.split_ratio.get(), *self.split_options)
        self.split_dropdown.pack(pady=5)

        # Option: Train models sequentially
        ttk.Checkbutton(self.right_frame, text="Train Models Sequentially", variable=self.train_sequentially, command=self.toggle_training_mode).pack(pady=5)

        # A frame that will contain either single model training options or sequential training UI.
        self.training_mode_frame = ttk.Frame(self.right_frame)
        self.training_mode_frame.pack(fill='both', expand=True)

        # Single Model Training UI
        self.single_model_frame = ttk.Frame(self.training_mode_frame)
        self.single_model_frame.pack(fill='both', expand=True)
        ttk.Label(self.single_model_frame, text="Select Model:").pack(pady=5)
        self.model_dropdown = ttk.OptionMenu(self.single_model_frame, self.model_name, self.model_name.get(), *self.models.keys())
        self.model_dropdown.pack(pady=5)
        ttk.Label(self.single_model_frame, text="Epochs:").pack(pady=5)
        ttk.Entry(self.single_model_frame, textvariable=self.epochs).pack(pady=5)
        ttk.Label(self.single_model_frame, text="Batch Size:").pack(pady=5)
        ttk.Entry(self.single_model_frame, textvariable=self.batch_size).pack(pady=5)
        ttk.Label(self.single_model_frame, text="Learning Rate:").pack(pady=5)
        ttk.Entry(self.single_model_frame, textvariable=self.learning_rate).pack(pady=5)
        ttk.Checkbutton(self.single_model_frame, text="Fine-tune Model", variable=self.fine_tune).pack(pady=5)
        ttk.Checkbutton(self.single_model_frame, text="Use Cross Validation", variable=self.use_cross_validation).pack(pady=5)
        ttk.Label(self.single_model_frame, text="Number of Folds:").pack(pady=5)
        ttk.Entry(self.single_model_frame, textvariable=self.num_folds).pack(pady=5)
        ttk.Button(self.single_model_frame, text="Train Model", command=self.train_model).pack(pady=10)

        # Sequential Training UI (hidden by default)
        self.sequential_model_frame = ttk.Frame(self.training_mode_frame)
        # Initially hidden:
        self.sequential_model_frame.pack_forget()
        ttk.Label(self.sequential_model_frame, text="Select Models for Sequential Training:").pack(pady=5)
        self.seq_listbox = Listbox(self.sequential_model_frame, selectmode=MULTIPLE, height=6)
        for model_key in self.models.keys():
            self.seq_listbox.insert(END, model_key)
        self.seq_listbox.pack(pady=5, fill='x')
        ttk.Button(self.sequential_model_frame, text="Train Selected Models Sequentially", command=self.train_models_sequentially).pack(pady=10)

        # Bottom panel: Log, Timer, and Progress Bar
        self.bottom_frame = ttk.Frame(root)
        self.bottom_frame.pack(side='bottom', fill='x', padx=10, pady=10)
        self.timer_label = ttk.Label(self.bottom_frame, text="Timer: 00:00:00")
        self.timer_label.pack(side='left', padx=5)
        self.progress_bar = ttk.Progressbar(self.bottom_frame, orient='horizontal', length=200, mode='determinate')
        self.progress_bar.pack(side='left', padx=5)
        self.log_text = tk.Text(self.bottom_frame, height=5)
        self.log_text.pack(fill='x', padx=5, pady=5)

    # ---------------------------
    # UI Helper Methods
    # ---------------------------
    def toggle_test_dataset(self):
        if self.use_separate_test.get():
            self.test_dataset_button.pack(pady=5)
            self.test_dataset_label.pack(pady=5)
        else:
            self.test_dataset_button.pack_forget()
            self.test_dataset_label.pack_forget()

    def toggle_training_mode(self):
        if self.train_sequentially.get():
            self.single_model_frame.pack_forget()
            self.sequential_model_frame.pack(fill='both', expand=True)
        else:
            self.sequential_model_frame.pack_forget()
            self.single_model_frame.pack(fill='both', expand=True)

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(END, f"{timestamp} - {message}\n")
        self.log_text.see(END)

    def update_queue_status(self, model, status):
        # Update the Treeview for the given model (iid is the model name)
        self.queue_tree.set(model, "Status", status)

    # ---------------------------
    # Dataset Selection
    # ---------------------------
    def select_dataset(self):
        path = filedialog.askdirectory()
        if path:
            self.dataset_path.set(path)

    def select_test_dataset(self):
        path = filedialog.askdirectory()
        if path:
            self.test_dataset_path.set(path)

    # ---------------------------
    # Model Building
    # ---------------------------
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

    def build_model(self, model_name, input_shape, num_classes, fine_tune):
        if model_name == "BasicCNN":
            return self.build_basic_cnn(input_shape, num_classes)
        else:
            base_model = self.models[model_name](weights='imagenet', include_top=False, input_shape=input_shape)
            if not fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(128, activation="relu")(x)
            predictions = Dense(num_classes, activation="softmax")(x)
            return Model(inputs=base_model.input, outputs=predictions)

    # ---------------------------
    # Training Methods
    # ---------------------------
    def train_model(self):
        if not self.dataset_path.get():
            self.log("Please select a training dataset folder.")
            return
        # Start training in a new thread (single model training)
        training_thread = threading.Thread(target=self._train_model_thread)
        training_thread.start()

    def _train_model_thread(self, model_name_override=None):
        # Use the override if provided (for sequential training)
        model_name = model_name_override if model_name_override is not None else self.model_name.get()
        epochs = self.epochs.get()
        batch_size = self.batch_size.get()
        learning_rate = self.learning_rate.get()
        fine_tune = self.fine_tune.get()

        # Check for cross validation option
        if self.use_cross_validation.get():
            self.perform_cross_validation(model_name, epochs, batch_size, learning_rate, fine_tune)
            return

        self.log("Preparing data for model: " + model_name)

        # Choose image size based on model requirements
        if model_name == "BasicCNN":
            img_size = (64, 64)
        elif model_name in ["Xception", "InceptionV3"]:
            img_size = (299, 299)
        else:
            img_size = (224, 224)

        # Use the chosen split ratio
        val_split = self.split_mapping.get(self.split_ratio.get(), 0.2)
        datagen = ImageDataGenerator(
            rescale=1.0/255,
            validation_split=val_split,
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
        self.log("Training model: " + model_name)
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            verbose=1,
            callbacks=[self.TrainingCallback(self.root, epochs), early_stopping]
        )

        # Save the model along with training details
        self.log("Training complete for model: " + model_name + ". Saving model and training details...")
        model_dir = r"C:\Users\User\OneDrive\Documents\SignLanguageApp\TrainedBinary5Model"
        os.makedirs(model_dir, exist_ok=True)
        model_file_name = f"{model_name}_model.h5"
        model_save_path = os.path.join(model_dir, model_file_name)
        try:
            model.save(model_save_path)
            self.log("Model saved successfully to: " + model_save_path)
        except Exception as e:
            self.log("Error saving model: " + str(e))
            return

        training_details = {
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "fine_tune": fine_tune,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_val_split": f"Training ({100-int(val_split*100)}%) | Validation ({int(val_split*100)}%)",
            "training_history": history.history
        }
        details_file_name = f"{model_name}_training_details.json"
        details_save_path = os.path.join(model_dir, details_file_name)
        try:
            with open(details_save_path, "w") as details_file:
                json.dump(training_details, details_file, indent=4)
            self.log("Training details saved successfully to: " + details_save_path)
        except Exception as e:
            self.log("Error saving training details: " + str(e))
            return

        results_folder = os.path.join(os.getcwd(), f"{model_name}_results")
        os.makedirs(results_folder, exist_ok=True)
        self.plot_training_history(history, results_folder, model_name)
        # If a separate test dataset is selected, use it for evaluation; otherwise use the training dataset.
        test_path = self.test_dataset_path.get() if self.use_separate_test.get() and self.test_dataset_path.get() else self.dataset_path.get()
        self.evaluate_and_save_metrics(model, validation_generator, model_name, test_path, results_folder)

        # ---------------------------
        # Generate and save Grad-CAM heatmap for a sample training image
        # ---------------------------
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
                else:
                    self.log("No image found in the sample class folder for Grad-CAM.")
            else:
                self.log("No class folders found in the dataset directory.")
        except Exception as e:
            self.log("Error generating Grad-CAM heatmap: " + str(e))

    def perform_cross_validation(self, model_name, epochs, batch_size, learning_rate, fine_tune):
        self.log("Performing cross validation for model: " + model_name)
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
            self.log(f"Training fold {fold_idx}/{self.num_folds.get()}...")
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
        
        self.log("Cross validation complete. Results saved.")

        self.log("Training final model on entire dataset after CV...")
        final_datagen = ImageDataGenerator(
            rescale=1.0/255,
            validation_split=val_split,
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False
        )
        final_train_generator = final_datagen.flow_from_directory(
            self.dataset_path.get(),
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        final_validation_generator = final_datagen.flow_from_directory(
            self.dataset_path.get(),
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        num_classes = len(final_train_generator.class_indices)
        final_model = self.build_model(model_name, (*img_size, 3), num_classes, fine_tune)
        final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        final_early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        final_history = final_model.fit(
            final_train_generator,
            validation_data=final_validation_generator,
            epochs=epochs,
            verbose=1,
            callbacks=[final_early_stopping]
        )

        final_model_dir = r"C:\Users\User\OneDrive\Documents\SignLanguageApp\TrainedBinary5Model_CV"
        os.makedirs(final_model_dir, exist_ok=True)
        final_model_file_name = f"{model_name}_final_model.h5"
        final_model_save_path = os.path.join(final_model_dir, final_model_file_name)
        try:
            final_model.save(final_model_save_path)
            self.log("Final model saved successfully to: " + final_model_save_path)
        except Exception as e:
            self.log("Error saving final model: " + str(e))
            return

        final_training_details = {
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "fine_tune": fine_tune,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_val_split": f"Training ({100-int(val_split*100)}%) | Validation ({int(val_split*100)}%)",
            "final_training_history": final_history.history,
            "cross_validation_results": {
                "overall_accuracy": overall_accuracy,
                "overall_precision": overall_precision,
                "overall_recall": overall_recall,
                "overall_f1": overall_f1,
                "classification_report": report
            }
        }
        final_details_file_name = f"{model_name}_final_training_details.json"
        final_details_save_path = os.path.join(final_model_dir, final_details_file_name)
        try:
            with open(final_details_save_path, "w") as f:
                json.dump(final_training_details, f, indent=4)
            self.log("Final training details saved successfully to: " + final_details_save_path)
        except Exception as e:
            self.log("Error saving final training details: " + str(e))
            return

        self.log(f"Final model and training details saved in {final_model_dir}")

    def train_models_sequentially(self):
        selected_indices = self.seq_listbox.curselection()
        if not selected_indices:
            self.log("No models selected for sequential training.")
            return
        selected_models = [self.seq_listbox.get(i) for i in selected_indices]
        total_models = len(selected_models)
        self.progress_bar["maximum"] = total_models
        self.progress_bar["value"] = 0

        # Clear and populate the sequential training queue
        for item in self.queue_tree.get_children():
            self.queue_tree.delete(item)
        for model in selected_models:
            self.queue_tree.insert("", "end", iid=model, values=(model, "Pending"))

        # Start a timer thread
        start_time = datetime.now()
        def update_timer():
            while self.progress_bar["value"] < total_models:
                elapsed = datetime.now() - start_time
                self.timer_label.config(text=f"Timer: {str(elapsed).split('.')[0]}")
                time.sleep(1)
        timer_thread = threading.Thread(target=update_timer, daemon=True)
        timer_thread.start()

        def sequential_training():
            for idx, model in enumerate(selected_models, start=1):
                self.update_queue_status(model, "In Progress")
                try:
                    self._train_model_thread(model_name_override=model)
                    self.update_queue_status(model, "Completed")
                    self.log(f"Model {model} trained successfully.")
                except Exception as e:
                    self.update_queue_status(model, f"Error: {e}")
                    self.log(f"Error training model {model}: {e}")
                self.progress_bar["value"] = idx
            self.log("Sequential training complete.")
        threading.Thread(target=sequential_training, daemon=True).start()

    def generate_and_save_gradcam_heatmap(self, model, sample_img_path, img_size, model_name, results_folder):
        from tensorflow.keras.preprocessing import image
        # Load and preprocess the image
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
        self.log("Grad-CAM heatmap saved to: " + output_path)

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

    class TrainingCallback(tf.keras.callbacks.Callback):
        def __init__(self, root, total_epochs):
            super().__init__()
            self.total_epochs = total_epochs
            self.root = root

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            accuracy = logs.get('accuracy', 0)
            val_accuracy = logs.get('val_accuracy', 0)
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            text = (f"Epoch {epoch + 1}/{self.total_epochs}\n"
                    f"Training Accuracy: {accuracy:.4f}, Training Loss: {loss:.4f}\n"
                    f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")
            # For demonstration, we print the text.
            self.root.after(0, lambda: print(text))

# ---------------------------
# Main Block
# ---------------------------
if __name__ == "__main__":
    print("Available GPU devices:", tf.config.list_physical_devices('GPU'))
    root = Tk()
    app = ASLTrainerApp(root)
    root.mainloop()

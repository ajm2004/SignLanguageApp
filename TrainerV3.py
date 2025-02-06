import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNet, MobileNetV2, ResNet50, EfficientNetB0, Xception, DenseNet121, VGG16, NASNetMobile, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tkinter import Tk, Label, Button, filedialog, StringVar, DoubleVar, IntVar, OptionMenu, Entry, Checkbutton, BooleanVar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import json
from datetime import datetime

# For evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

class ASLTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Image Trainer")
        self.root.geometry("500x800")
        self.root.minsize(400, 500)
        # Variables
        self.dataset_path = StringVar()
        self.model_name = StringVar(value="BasicCNN")
        self.epochs = IntVar(value=50)
        self.batch_size = IntVar(value=16)
        self.learning_rate = DoubleVar(value=0.001)
        self.fine_tune = BooleanVar(value=False)
        self.use_cross_validation = BooleanVar(value=False)
        self.num_folds = IntVar(value=5)
        
        self.models = {
            "BasicCNN": self.build_basic_cnn,
            "MobileNet": MobileNet,
            "MobileNetV2": MobileNetV2,
            "EfficientNetB0": EfficientNetB0,
            "ResNet50": ResNet50,
            "Xception": Xception,
            "DenseNet121": DenseNet121,
            "VGG16": VGG16,
            "NASNetMobile": NASNetMobile,
            "InceptionV3": InceptionV3
        }

        # UI Elements
        Label(root, text="ASL Image Trainer", font=("Arial", 16)).pack(pady=10)

        Button(root, text="Select Dataset", command=self.select_dataset).pack(pady=5)
        Label(root, textvariable=self.dataset_path).pack(pady=5)

        Label(root, text="Select Model:").pack(pady=5)
        OptionMenu(root, self.model_name, *self.models.keys()).pack(pady=5)

        Label(root, text="Epochs:").pack(pady=5)
        Entry(root, textvariable=self.epochs).pack(pady=5)

        Label(root, text="Batch Size:").pack(pady=5)
        Entry(root, textvariable=self.batch_size).pack(pady=5)

        Label(root, text="Learning Rate:").pack(pady=5)
        Entry(root, textvariable=self.learning_rate).pack(pady=5)

        Checkbutton(root, text="Fine-tune Model", variable=self.fine_tune).pack(pady=5)
        
        # New cross validation UI options:
        Checkbutton(root, text="Use Cross Validation", variable=self.use_cross_validation).pack(pady=5)
        Label(root, text="Number of Folds:").pack(pady=5)
        Entry(root, textvariable=self.num_folds).pack(pady=5)

        self.train_button = Button(root, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        self.status_label = Label(root, text="Status: Waiting for input...", wraplength=400, justify="left")
        self.status_label.pack(pady=10)

        self.progress_label = Label(root, text="", wraplength=400, justify="left")
        self.progress_label.pack(pady=10)

    def select_dataset(self):
        path = filedialog.askdirectory()
        if path:
            self.dataset_path.set(path)

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

    def train_model(self):
        if not self.dataset_path.get():
            self.status_label.config(text="Please select a dataset folder.")
            return
        # Start the training process in a new thread
        training_thread = threading.Thread(target=self._train_model_thread)
        training_thread.start()

    def _train_model_thread(self):
        model_name = self.model_name.get()
        epochs = self.epochs.get()
        batch_size = self.batch_size.get()
        learning_rate = self.learning_rate.get()
        fine_tune = self.fine_tune.get()

        # If cross validation is selected, use that branch.
        if self.use_cross_validation.get():
            self.perform_cross_validation(model_name, epochs, batch_size, learning_rate, fine_tune)
            return

        self.status_label.config(text="Preparing data...")

        # Choose image size based on model requirements
        img_size = (64, 64) if model_name == "BasicCNN" else (299, 299) if model_name in ["Xception", "InceptionV3"] else (224, 224)
        datagen = ImageDataGenerator(
            rescale=1.0/255,
            validation_split=0.2,
            rotation_range=15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
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
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        self.status_label.config(text="Training model...")

        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            verbose=1,
            callbacks=[self.TrainingCallback(self.progress_label, epochs), early_stopping]
        )

        # Save the model along with training details
        self.status_label.config(text="Training complete. Saving model and training details...")

        # Set the folder path explicitly where the model is to be saved.
        model_dir = r"C:\Users\User\OneDrive\Documents\SignLanguageApp\TrainedBinary2Model"
        print("Attempting to create or access folder:", model_dir)
        try:
            os.makedirs(model_dir, exist_ok=True)
            print("Folder created or already exists.")
        except Exception as e:
            print("Error creating folder:", e)
            self.status_label.config(text=f"Error creating folder: {e}")
            return

        model_file_name = f"{model_name}_model.h5"
        model_save_path = os.path.join(model_dir, model_file_name)
        print("Model will be saved to:", model_save_path)
        
        try:
            model.save(model_save_path)
            print("Model saved successfully.")
        except Exception as e:
            print("Error saving model:", e)
            self.status_label.config(text=f"Error saving model: {e}")
            return

        # Save training details (e.g., hyperparameters, training history, timestamp)
        training_details = {
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "fine_tune": fine_tune,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_val_split": "Training (80%) | Validation (20%)",
            "training_history": history.history  # This contains metrics for each epoch
        }
        details_file_name = f"{model_name}_training_details.json"
        details_save_path = os.path.join(model_dir, details_file_name)
        print("Saving training details to:", details_save_path)
        try:
            with open(details_save_path, "w") as details_file:
                json.dump(training_details, details_file, indent=4)
            print("Training details saved successfully.")
        except Exception as e:
            print("Error saving training details:", e)
            self.status_label.config(text=f"Error saving training details: {e}")
            return

        self.status_label.config(text=f"Model and training details saved in {model_dir}")

        # Continue with plotting training history and evaluation if needed
        results_folder = os.path.join(os.getcwd(), f"{model_name}_results")
        os.makedirs(results_folder, exist_ok=True)
        self.plot_training_history(history, results_folder, model_name)
        self.evaluate_and_save_metrics(model, validation_generator, model_name, self.dataset_path.get(), results_folder)

    def perform_cross_validation(self, model_name, epochs, batch_size, learning_rate, fine_tune):
        self.status_label.config(text="Performing cross validation...")
        # Get all image file paths and labels
        file_paths, labels, class_names = self.load_image_paths_and_labels(self.dataset_path.get())
        # Determine the required image size
        img_size = (64, 64) if model_name == "BasicCNN" else (299, 299) if model_name in ["Xception", "InceptionV3"] else (224, 224)
        
        # Define a function to load and preprocess images
        def load_and_preprocess_image(path, label):
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3)
            # Set the shape explicitly to inform TensorFlow of the three channels.
            image.set_shape([None, None, 3])
            image = tf.image.resize(image, img_size)
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        # Prepare KFold cross validation
        kf = KFold(n_splits=self.num_folds.get(), shuffle=True, random_state=42)
        all_y_true = []
        all_y_pred = []
        fold_metrics = []
        fold_idx = 1
        
        for train_index, val_index in kf.split(file_paths):
            self.status_label.config(text=f"Training fold {fold_idx}/{self.num_folds.get()}...")
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
            # For cross validation using our custom dataset, we use sparse labels.
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
        
        self.status_label.config(text="Cross validation complete. Results saved.")

        # ----- NEW: Train and Save a Final Model on the Entire Dataset -----
        self.status_label.config(text="Training final model on entire dataset after CV...")
        final_datagen = ImageDataGenerator(
            rescale=1.0/255,
            validation_split=0.2,
            rotation_range=15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
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

        # Save the final model and training details
        final_model_dir = r"C:\Users\User\OneDrive\Documents\SignLanguageApp\TrainedBinary2Model_CV"
        print("Attempting to create or access final model folder:", final_model_dir)
        try:
            os.makedirs(final_model_dir, exist_ok=True)
            print("Final model folder created or already exists.")
        except Exception as e:
            print("Error creating final model folder:", e)
            self.status_label.config(text=f"Error creating final model folder: {e}")
            return

        final_model_file_name = f"{model_name}_final_model.h5"
        final_model_save_path = os.path.join(final_model_dir, final_model_file_name)
        print("Final model will be saved to:", final_model_save_path)
        try:
            final_model.save(final_model_save_path)
            print("Final model saved successfully.")
        except Exception as e:
            print("Error saving final model:", e)
            self.status_label.config(text=f"Error saving final model: {e}")
            return

        final_training_details = {
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "fine_tune": fine_tune,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_val_split": "Training (80%) | Validation (20%)",
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
        print("Saving final training details to:", final_details_save_path)
        try:
            with open(final_details_save_path, "w") as f:
                json.dump(final_training_details, f, indent=4)
            print("Final training details saved successfully.")
        except Exception as e:
            print("Error saving final training details:", e)
            self.status_label.config(text=f"Error saving final training details: {e}")
            return

        self.status_label.config(text=f"Final model and training details saved in {final_model_dir}")

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
        """
        Scans the dataset folder (which contains subdirectories for each class)
        and returns arrays for file paths, labels, and a list of class names.
        """
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
        def __init__(self, progress_label, total_epochs):
            super().__init__()
            self.progress_label = progress_label
            self.total_epochs = total_epochs

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            accuracy = logs.get('accuracy', 0)
            val_accuracy = logs.get('val_accuracy', 0)
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            text = (f"Epoch {epoch + 1}/{self.total_epochs}\n"
                    f"Training Accuracy: {accuracy:.4f}, Training Loss: {loss:.4f}\n"
                    f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")
            self.progress_label.after(0, self.progress_label.config, {'text': text})

if __name__ == "__main__":
    print("Available GPU devices:", tf.config.list_physical_devices('GPU'))
    root = Tk()
    app = ASLTrainerApp(root)
    root.mainloop()

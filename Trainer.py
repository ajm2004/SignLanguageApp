import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNet, MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tkinter import Tk, Label, Button, filedialog, StringVar, DoubleVar, IntVar, OptionMenu, Entry, Checkbutton, BooleanVar
import matplotlib.pyplot as plt
import threading

class ASLTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Image Trainer")
        self.root.geometry("500x700")
        self.root.minsize(400, 500)
        # Variables
        self.dataset_path = StringVar()
        self.model_name = StringVar(value="MobileNetV1")
        self.epochs = IntVar(value=50)
        self.batch_size = IntVar(value=16)
        self.learning_rate = DoubleVar(value=0.001)
        self.fine_tune = BooleanVar(value=False)
        self.models = {
            "MobileNetV1": MobileNet,
            "MobileNetV2": MobileNetV2,
            "EfficientNetB0": EfficientNetB0,
            "ResNet50": ResNet50
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

    def build_model(self, base_model_name, fine_tune):
        base_model = self.models[base_model_name](weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        if not fine_tune:
            for layer in base_model.layers:
                layer.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        predictions = Dense(len(os.listdir(self.dataset_path.get())), activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

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

        self.status_label.config(text="Preparing data...")

        datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

        train_generator = datagen.flow_from_directory(
            self.dataset_path.get(),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = datagen.flow_from_directory(
            self.dataset_path.get(),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )

        self.status_label.config(text="Building model...")
        model = self.build_model(model_name, fine_tune)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.status_label.config(text="Training model...")

        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            verbose=1,
            callbacks=[self.TrainingCallback(self.progress_label, epochs)]
        )

        self.status_label.config(text="Training complete. Saving model...")

        save_path = os.path.join(self.dataset_path.get(), f"{model_name}_model.h5")
        model.save(save_path)

        self.status_label.config(text=f"Model saved at {save_path}")

        self.plot_training_history(history)

    class TrainingCallback(tf.keras.callbacks.Callback):
        def __init__(self, progress_label, total_epochs):
            super().__init__()
            self.progress_label = progress_label
            self.total_epochs = total_epochs

        def on_epoch_end(self, epoch, logs=None):
            accuracy = logs.get('accuracy', 0)
            val_accuracy = logs.get('val_accuracy', 0)
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)

            # Update the progress label with training progress
            self.progress_label.after(0, self.progress_label.config, {
                'text': f"Epoch {epoch + 1}/{self.total_epochs}\n"
                        f"Training Accuracy: {accuracy:.4f}, Training Loss: {loss:.4f}\n"
                        f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}"
            })

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    root = Tk()
    app = ASLTrainerApp(root)
    root.mainloop()

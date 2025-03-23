# Sign Language Application and Deep Learning Study

This repository focuses on developing and executing a sign language recognition application (ASL fingerspelling) based on two types of pipelines (Binary based and Landmarked based). The process followed consists of capturing images for dataset, feature-engineering and augmenting images then using it for training and evaluating each models based on their accuracy (primary metrics) and using best models in real-time testing using a GUI based application. 

## User Guide:

### A. Data Capturing (In dataCapture.ipynb):

#### Pipeline Descriptions

1. **Pipeline 2: With Landmarks - Pipeline_2**

- Uses both MediaPipe segmentation and hand landmarks detection.
- Captures video from the webcam and uses a hand detector to identify the hand.
- After detecting the hand, computes an adjustable crop with padding around the hand.
- Applies MediaPipe Selfie Segmentation to remove the background from the cropped image.
- Overlays hand landmarks and connections on the segmented image.
- Converts the overlaid segmented image into a binary format.
- Resizes the processed image to a standardized size and saves the image once collection is activated.
- Designed to aid in visualizing hand structure with landmark annotations during data capture.
  **Execution-** Run the cell: With Landmarks - Pipeline_2 and perform image collection.

2. **Pipeline 1: Just Binary - Pipeline_1**

- Similar initial steps with webcam capture, hand detection, and adjustable cropping.
- Applies MediaPipe segmentation to remove background without adding landmark overlays.
- Directly converts the segmented crop into a binary (black and white) image.
- Resizes the binary image for consistency and saves the image when collection is activated.
- Focuses purely on obtaining a clean binary silhouette without additional markings.
  **Execution-** Run the cell: Just Binary - Pipeline_1 and perform image collection.

---

### B. Data Augmentation and Joining (In dataAug.ipynb)

This Jupyter Notebook performs two main tasks:

1. **Data Augmentation**

   - The code imports the necessary modules (e.g., os, random, PIL, numpy) and defines image augmentation functions:
     - **random_rotation:** Rotates an image by a random angle between -20° to 20°.
     - **random_flip:** Randomly mirrors an image horizontally.
   - The `augment_image` function applies these two augmentations.
   - The notebook sets up input and output directories. Images from the input folder are converted to grayscale, augmented using the defined functions, and then saved into a specified output folder (preserving the folder structure).
2. **Joining Augmented Data with Original Data**

   - The code then imports additional modules for file management.
   - Directories for the original images and the augmented images are defined.
   - A merge function walks through the folders of both datasets. It copies the images to a combined output folder:
     - Files from the original dataset remain unchanged.
     - Files from the augmented dataset are renamed by adding a `_AUG` suffix to differentiate them.
   - The merged dataset is stored in a new output directory.

##### How to Assign Files and Run the Code

1. **Set Directories:**

   - Update the paths in the code cells (for example, `input_folder`, `output_folder`, `dataset1`, `dataset2`, and `output_dataset`) to point to the locations where your images and datasets are stored.
2. **Run the Notebook Cells Sequentially:**

   - Execute the cells in order. The augmentation cells should be run first so that the output folder is populated with augmented images.
   - Finally, run the cell that merges the original and augmented datasets.

By following these steps, you can customize the file paths and execute the notebook to process and merge your datasets.

---

### C. Training models on collected dataset (In TrainerV5_1.py)

#### Overview

This Python script implements a graphical user interface (GUI) application meant (but not limited) for training American Sign Language (ASL) image classifiers. It uses [TensorFlow](https://www.tensorflow.org/) and its Keras API to build and train various deep learning models. The application includes the following features:

- **Model Selection:** Choose from multiple pretrained architectures (e.g., MobileNet, VGG16, ResNet50, EfficientNetB0, etc.) or a custom BasicCNN and a Fusion Model.
- **Image Data Handling:** Utilizes `ImageDataGenerator` for data augmentation and splitting the dataset into training and validation subsets.
- **Training Management:** Supports setting parameters like epochs, batch size, learning rate, and fine-tuning options. It also includes an option for cross-validation.
- **Real-Time Monitoring:** Displays training progress, estimated remaining time, elapsed time, and logs each training epoch’s metrics.
- **Evaluation and Visualization:** Generates training plots, confusion matrices, evaluation metrics, and applies Grad-CAM to produce heatmaps for visualizing model attention.
- **User-Friendly GUI:** Built with Tkinter, the application allows users to select dataset directories, model save locations, and configure training parameters through an intuitive interface.

#### Dependencies and Installation

To run this application, you will need to install several Python packages. It is recommended to use a virtual environment to manage dependencies. Below is a list of required packages along with installation instructions using `pip`:

1. **TensorFlow (with Keras integrated):**
   *Note: Ensure you have a version of TensorFlow that supports your hardware (GPU/CPU).*
   ```bash
   pip install tensorflow
   ```




2. **NumPy:**
   ```bash
   pip install numpy
   ```

3. **scikit-learn:**
   ```bash
   pip install scikit-learn
   ```

4. **Matplotlib:**
   ```bash
   pip install matplotlib
   ```

5. **Seaborn:**
   ```bash
   pip install seaborn
   ```

6. **OpenCV-Python:**
   ```bash
   pip install opencv-python
   ```

7. **Pillow:**
    ```bash
   pip install pillow
   ```

8. **Tkinter:**
   *Tkinter is usually bundled with Python. If it is not available, refer to your OS-specific instructions.*


#### Hardware Recommendations

##### GPU
A dedicated GPU (such as NVIDIA with CUDA support) is highly recommended to significantly speed up training, especially when working with larger architectures like Xception or InceptionV3.

##### CPU
A strong multi-core CPU is beneficial for data preprocessing and handling the GUI, especially if a GPU is not available.

##### RAM
At least 8 GB of RAM is recommended. More may be required if your dataset is large.

##### Storage
Ensure you have sufficient disk space for saving models, training logs, and visualizations.

#### How to Run and Use the Application

##### Running the Script

Execute the script from your terminal or command prompt:

```bash
python TrainerV5_1.py
```

The application window should appear after a short initialization period.

#### Using the GUI

##### Select Dataset

- **Click the "Select Dataset" button** to choose the folder containing your ASL images.
- The images should be organized in subdirectories by class.

##### Select Save Directory

- **Use the "Select Save Directory" button** to specify where the trained models and training details should be saved.

##### Model Selection

- **Choose a Model:**  
  Select a model from the dropdown menu.

- **Add Model:**  
  Click **"Add Model"** to add it to the training sequence (up to a maximum of 7 models).

- **Remove Model:**  
  To remove a model from the sequence, select it in the treeview and click **"Remove Selected Model"**.

##### Set Training Parameters

Adjust parameters such as:

- **Train/Test Split**
- **Epochs**
- **Batch Size**
- **Learning Rate**
- **Fine-Tuning Option**

Optionally, you can enable **Cross Validation** and set the **Number of Folds**.

##### Start Training

- **Click the "Train Models" button** to start the training process.
- Training progress, including epoch-wise metrics and estimated remaining time, will be displayed.

##### Interrupt Training

- If needed, **click "Stop Training"** to interrupt the ongoing training process.

##### Outputs and Visualizations

After training, the application saves:

- **Trained Model Files:**  
  Saved in `.h5` format.

- **Training Details:**  
  Saved in JSON format.

- **Evaluation Reports and Plots:**  
  Including training metrics plots, confusion matrices, and Grad-CAM heatmaps, saved in designated result folders.


---

### D. Real-Time Testing (Using ASL_RecogAPP_V3_3.py):

This application provides real-time American Sign Language fingerspelling recognition using computer vision and deep learning. It leverages several libraries such as OpenCV, MediaPipe, and cvzone to detect and track hand gestures and then classify them into letters. The user interface is built with Tkinter and displays the camera feed, binary images, hand landmarks, prediction details, performance metrics, and various controls.



#### Overview

- **Real-Time Hand Detection:** Uses OpenCV and MediaPipe for capturing video and detecting hand landmarks.
- **Classification Models:** 
  - **Pipeline-1 (T1):** Works with a simpler binary image processing pipeline.
  - **Pipeline-2 (T2):** Uses a more advanced landmark-based approach.
- **Ensemble Mode:** Allows combining predictions from multiple models.
- **User Interface:** A rich Tkinter-based GUI displaying:
  - A main camera feed.
  - Side panels for performance metrics, resource usage, logs, and control settings.
  - A text area to accumulate recognized letters into words.
- **Additional Features:**
  - Auto-type functionality that automatically enters letters after a short delay.
  - Text-to-speech pronunciation using `pyttsx3`.
  - Logging of performance metrics (FPS, prediction rates, etc.) and system resource information.

#### Dependencies and Installation

##### Required Python Packages

Make sure you have Python 3.x installed. The following libraries are needed:

- **OpenCV:** For video capture and image processing.
- **MediaPipe:** For hand tracking.
- **cvzone:** For hand detection and model classification modules.
- **numpy:** For numerical operations.
- **Pillow:** For image processing within Tkinter.
- **pyttsx3:** For text-to-speech functionality.
- **psutil:** For retrieving system resource information.
- **GPUtil:** (Optional) For GPU resource info.
- **tkinter:** Usually comes with Python; if not, install via your package manager.

##### Installation Commands

You can install the required packages using `pip`:

```bash
pip install opencv-python mediapipe cvzone numpy pillow pyttsx3 psutil GPUtil
```
*Note: If GPUtil is not required or causes issues, the script will still work without it.*


### Setup and Running the Application



##### Place the Trained Models

- The script contains a dictionary (`model_options`) mapping model names to their respective file paths.
- You can use the provided model paths or replace them with the paths to your own trained models.
- **Note:** Models labeled with **(T1)** are for Pipeline-1 (binary image processing), while those with **(T2)** are for Pipeline-2 (landmark based).

### Run the Application

1. Open a terminal in the directory where the script is located.
2. Run the script with:

   ```bash
   python ASL_RecogAPP_V3_3.py
   ```
3. A splash screen will appear for a few seconds before the main UI is shown.



### Using the Application - Real-Time Recognition::

**Camera Feed**
- The main panel displays your camera feed with hand detection overlays.

**Hand Crop & Processing**
- The application crops the hand area and applies either segmentation or landmark overlays depending on the selected pipeline.

**Prediction Display**
- A prediction box appears near the hand, showing the recognized letter along with its confidence score.

#### Input and Auto Type:

**Manual Input**
- **Space Bar:** Press to enter the predicted letter or a space if no prediction is available.
- **Backspace:** Deletes the last character from the current word.

**Auto-Type Mode**
- Enable auto-type from the settings to automatically add predicted letters after a short delay (default 1.5 seconds).

#### Text-to-Speech:

**Pronunciation**
- **Pronounce Button / Enter Key:**  
  Use the "Pronounce" button or press Enter (when not editing text) to have the current word pronounced using the selected accent.
- The accent can be chosen from the settings menu.

#### Changing Models and Settings:

**Settings Menu:**

  1. #####  Access
  - Access the settings via the "Settings" button in the right panel.

  2. #####  Options
  - **Model Selection:**  
  Choose a different model from the drop-down list. For custom models, update the file path in the `model_options` dictionary.
  - **Segmentation Filter:**  
  Use if you are in a bright room.
  - **Auto Type:**  
  Automatically input letters based on prediction.
  - **Ensemble Mode:**  
  Combine predictions from multiple models (note: this increases resource usage).

*Pipeline Selection: Remember that models with **(T1)** are for a simpler pipeline, whereas **(T2)** models use a landmark-based pipeline.*


#### Performance Metrics and Logs

1. #### Left Panel
- Shows performance metrics including FPS, prediction rate, and system resource usage (CPU, GPU, Memory).

2. #### Logging
- Performance data is logged to a file in the `AppFPS_Results` folder for later review.

---

## Conclusion

This project integrates multiple components to create a robust sign language application. It begins with advanced data capturing pipelines that offer both landmark-based and binary image processing. The data augmentation and merging steps enrich the dataset, ensuring a diverse training set. A user-friendly GUI facilitates training deep learning models with customizable parameters and real-time monitoring of progress. Finally, the application enables real-time sign language recognition, complete with features such as auto-type and text-to-speech, thereby delivering an end-to-end solution for sign language interpretation.

---

# Face Emotion Recognition

Face Emotion Recognition is an AI pipeline for detecting the face on the image and classifying it to belong to a certain set of emotions. 
The current branch is an Android mobile application that utilizes the proposed pipeline. 
It uses the [MTCNN](https://github.com/ipazc/mtcnn) (Multi-task Cascaded Convolutional Networks) model for face detection and the [EfficientNet](https://github.com/qubvel/efficientnet) model for recognizing emotions from facial expressions.
Moreover, the current branch provides the PyTorch Mobile version of a model as [SDK](https://github.com/Namadgi/face-emotion-recognition/blob/mobile/model_weights/fer.ptl), which takes an image as input and returns the label or the error code as output.

## App Installation

### Requirements

- Android: Android 7.0 (Nougat) or later
- Minimum 2GB RAM recommended

### Installation Steps

1. Clone this repository to your local machine.
   ```
   git clone https://github.com/Namadgi/face-emotion-recognition.git
   ```
2. Open the project in your preferred IDE (Android Studio for Android).
3. Build and run the project on your device or emulator.

![image](https://github.com/Namadgi/face-emotion-recognition/assets/44228198/53013beb-61e6-4372-9c6a-47eaf9d35968)

## SDK Usage

Depending on the OS and platform used, install the framework for PyTorch mobile inference...

## Credits

This project utilizes the following open-source libraries and pre-trained models:

- [Mobile-App](https://github.com/av-savchenko/face-emotion-recognition): The original repository.
- [MTCNN](https://github.com/ipazc/mtcnn): Multi-task Cascaded Convolutional Networks for face detection.
- [EfficientNet](https://github.com/qubvel/efficientnet): Efficient neural network architecture for emotion recognition.

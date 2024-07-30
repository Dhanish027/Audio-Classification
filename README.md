## Audio-Classification

The task of identifying what an audio represents is called audio classification. An audio classification model is trained to recognize various audio events. For example, you may train a model to recognize events representing three different events: clapping, finger snapping, and typing. TensorFlow Lite provides optimized pre-trained models that you can deploy in your mobile applications.

<p align="center">
    <img src='https://github.com/Dhanish027/Audio-Classification/blob/master/Audio1.png' height="200">
</p>

Audio Classification is a machine learning task that involves identifying and tagging audio signals into different classes or categories. The goal of audio classification is to enable machines to automatically recognize and distinguish between different types of audio, such as music, speech, and environmental sounds.


The following image shows the output of the audio classification model on Android.

<p align="center">
    <img src='https://github.com/Dhanish027/Audio-Classification/blob/master/Audio2.png' height="400">
</p>


## How it works

There are two versions of the YAMNet model converted to TFLite:

YAMNet Is the original audio classification model, with dynamic input size, suitable for Transfer Learning, Web and Mobile deployment. It also has a more complex output.

YAMNet/classification is a quantized version with a simpler fixed length frame input (15600 samples) and return a single vector of scores for 521 audio event classes.

## Inputs
The model accepts a 1-D float32 Tensor or NumPy array of length 15600 containing a 0.975 second waveform represented as mono 16 kHz samples in the range [-1.0, +1.0].

## Outputs
The model returns a 2-D float32 Tensor of shape (1, 521) containing the predicted scores for each of the 521 classes in the AudioSet ontology that are supported by YAMNet. The column index (0-520) of the scores tensor is mapped to the corresponding AudioSet class name using the YAMNet Class Map, which is available as an associated file yamnet_label_list.txt packed into the model file. See below for usage.





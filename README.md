# iLID
Automatic spoken language identification (LID) using deep learning.

## Motivation
We wanted to classify the spoken language within audio files, a process that usually serves as the first step for NLP or speech transcription.

We used two deep learning approaches using the Tensorflow and Caffe frameworks for different model configuration.

## Repo Structure

- **/Evaluation**
  - Prediction scripts for single audio files or list of files using Caffe
- **/Preprocessing**
  - Includes all scripts to convert a WAV audio file into spectrogram and mel-filter spectrogram images using a Spark Pipeline.
  - All scripts to create/extract the audio features
  - To convert a directory of WAV audio files using the Spark pipeline run: `./run.sh --inputPath {input_path} --outputPath {output_path} | tee sparkline.log -`
- **/model**
- **/tensorflow**
  - All the code for setting up and training various models with Tensorflow.
  - Includes training and prediction script. See `train.py` and `predict.py`.
  - Configure your learning parameters in `config.yaml`.
  - Add or change network under `/tensorflow/networks/instances/`.
- **/tools**
  - Some handy scripts to clean filenames, normalize audio files and other stuff.
- **/webserver**
  - A web demo to upload audio files for prediction.
  - See the included README
  

## Requirements
- Caffe 
- TensorFlow
- Spark
- Python 2.7
- OpenCV 2.4+

```
// Install additional Python requirements
pip install -r requirements.txt
```

## Model Training

For Caffe:
```
/models/{model_name}/training.sh
```

For Tensorflow:
```
/tensorflow/train.py
```

## Training Data
For training we used both the public [Voxforge](http://www.voxforge.org/) dataset and downloaded news reel videos from Youtube. Check out the training data repo: https://github.com/twerkmeister/iLID-data



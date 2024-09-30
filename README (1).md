
# Image Captioning Using Attention Models

This project implements an image captioning system using a deep learning approach that integrates Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) with an Attention Mechanism. The model is designed to generate captions for images by combining image feature extraction with sequence generation, using the power of TensorFlow and Keras libraries.


## Key Features:

- Import Libraries:

  The required libraries for image processing, deep learning, and text processing are imported, including TensorFlow, Keras, and NLTK.
- Data Preprocessing:

  The dataset used for this project is the Flickr 8k Dataset, which is sourced from Kaggle. It consists of 8,000 images and their corresponding captions. You can download the dataset from Kaggle's Flickr 8k page.

Model Architecture:

The InceptionV3 model is used for extracting image features.
The captions are tokenized and converted into sequences for feeding into the LSTM model.
The model is built with an encoder-decoder architecture using an LSTM network to generate captions based on image features.
Training:

The data is split into training and validation sets.
The model is trained to minimize the caption generation error using categorical cross-entropy loss.
Evaluation:

The model is evaluated using standard metrics to measure the quality of the generated captions.



## Project Structure

- Import Libraries:

The required libraries for image processing, deep learning, and text processing are imported, including TensorFlow, Keras, and NLTK.

- Data Preprocessing:

The dataset consists of images and their corresponding captions.
The function load_captions reads and processes the caption data from a text file. The file should contain image IDs and captions, which are paired together.
Images are preprocessed using the function preprocess_image to resize them to the required dimensions and apply necessary transformations for the model input.

- Model Architecture:

The InceptionV3 model is used for extracting image features.
The captions are tokenized and converted into sequences for feeding into the LSTM model.
The model is built with an encoder-decoder architecture using an LSTM network to generate captions based on image features.

- Training:

The data is split into training and validation sets.
The model is trained to minimize the caption generation error using categorical cross-entropy loss.

- Evaluation:

The model is evaluated using standard metrics to measure the quality of the generated captions.
## Requirements

- Python 3.7 or higher
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- PIL
- nltk
- tqdm
- scikit-learn
## How to Run

- Clone the repository or download the notebook.
- Install the necessary Python libraries using pip:

```bash
 pip install -r requirements.txt
```
    
## How to Run

- Clone the repository or download the notebook.
- Install the necessary Python libraries using pip:
- Download the Flickr 8k Dataset from Kaggle and extract the images into the ```Images/``` folder.
- Place the captions file (```captions.txt```) in the same directory as described in the notebook.
## Dataset
- The dataset should consist of images and a corresponding text file containing image IDs and captions. The image files are located in a directory specified by ```images_dir```, and the captions are loaded using the ```load_captions``` function.
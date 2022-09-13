# Food-101 Project
## What's Food-101 ?

Milestone Food-101 is an end-to-end **CNN Image Classification Model** which identifies the food in your image. 
It can identify over 100 different food classes
It is based upom a pre-trained Image Classification Model that comes with Keras and then retrained on the infamous **Food101 Dataset**.

* **Accuracy :** **`82%`**
* **Model :** **`EfficientNetB0`**
* **Dataset :** **`Food101`**

* This project is part of the Zero to Mastery Tensorflow Developer course (MileStone Project 1).
* This project based on the [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) Paper which used Convolutional Neuranetwork trained for 2 to 3 days to achieve 77.4% top-1 accuracy.
* The project is made by download the food101 dataset from the [TensorFlow dataset](https://www.tensorflow.org/datasets/catalog/food101)(size: 4.6GB) which consists of 750 images x 101 training classes = 75750 training images.
* I used the [EfficientNetB0](https://github.com/helloitsdaksh/Tensorflow_colab/blob/main/07_food_vision_milestone_project_1.ipynb) model with fine-tune freezed layers of the model.

## Since the resources on streamlit are limited you can run this on your local machine using following commands:
* clone the repo first
* `pip install -r requirements.txt`
* `streamlit run app.py`

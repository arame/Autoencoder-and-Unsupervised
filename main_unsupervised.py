
# # Unsupervised Learning
# 
# In the class we learned two commonly used unsupervised learning algorithms:
# * Principal Component Analysis
# * K-Means
# 
# In this exercise we will use the two to perform clustering for Iris Data set.
# 
# ## Iris Dataset
# 
# [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) is a very popular dataset in machine learning community. Developed by Fisher, it containd 3 classes each with 50 instances. Each of the three class refers to a type of Iris plant. Each data point consists of four different attributes and a class label:
# 
# 1. sepal length in cm
# 2. sepal width in cm
# 3. petal length in cm
# 4. petal width in cm
# 5. class:
#  * Iris Setosa
#  * Iris Versicolour
#  * Iris Virginica
# 
# 
# 
#@title Import Necessary Modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from display_data import barcharts, scatter_species
def main():
    # ### Exercise 1: Download Data
    # 
    # Download the data using `tf.keras.utils.get_file()` function. Pass two arguments to the function the file name and the url containg the data.
    # 
    # * For training data use the file name (`fname`) "iris_training.csv" and the url is "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
    # 
    # * For test data use the file name (`fname`) "iris_test.csv" and the url is "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
    # 
    # For more information and usage you can refer to [TensorFlow documentation](https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file).
    # 
    # 

    # To Do Complete the code
    ## Replace the ... with right code
    train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
    test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

    # ----------------- Do not change anything below ------------------------------------#
    CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
    SPECIES = ['Setosa', 'Versicolor', 'Virginica']

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

    # ### Exercise 2: Data Analysis and Visualization
    # 
    # Analyze the data. Analyzing data is avery important skill. You can start from simple information like size of test and train dataset. To random sample of few to plotting between different features. [Here](https://www.kaggle.com/kstaud85/iris-data-visualization) is an example dataset visualization notebook for Iris Dataset. You can take some ideas from here.
    # 
    # As part of the exercise we expect you to perform atleast three analysis/visualizations. 
    # You can use as many code cells as you need.
    # 
    # Remember `train` and `test` are Pandas DataFrames.

    ## To do write code to analyze and visualize the data
    #  you can add as many code cells as you requre
    # below for example we show the top 5 samples from training dataset.
    #train.head(5)
    scatter_species("Sepal Length by Width", train, "SepalLength", "SepalWidth", "Length", "Width", "sepal")
    scatter_species("Petal Length by Width", train, "PetalLength", "PetalWidth", "Length", "Width", "petal")
    barcharts("Sepal Length per species", train["Species"], train["SepalLength"], "Species", "Sepal Length", "Unsupervised_SepalLength")
    barcharts("Sepal Width per species", train["Species"], train["SepalWidth"], "Species", "Sepal Width", "Unsupervised_SepalWidth")
    barcharts("Petal Length per species", train["Species"], train["PetalLength"], "Species", "Petal Length", "Unsupervised_PetalLength")
    barcharts("Petal Width per species", train["Species"], train["PetalWidth"], "Species", "Petal Width", "Unsupervised_PetalWidth")
    # ### Exercise 3 : Preprocess the data
    # 
    # Implement the following steps:
    # 
    # 1. Drop the label - Species, since we are doing Unsupervised learning we do not need labels.
    # 
    #  ###### You may want to save labels separately too verify if indeed your model has been able to cluster properly.
    # 
    # 2. For PCA it is good if the data has zero mean and variance of 1. To achieve this subtract mean and  divide by standard deviation.
    # 

    ## Drop the labels
    i = 0
    train.drop(...)
    test.drop(...)

    # Subtract mean from individual value and divide by standard deviation

    normalized_train=...
    normalized_test=...

    # ### Exercise 4: 
    # Compute the SUV matrices using TensorFlow `linalg()` function. Once you get SUV matrices convert S matrix to diagonal matrix  using `tf.linalg.diag()` 

    # Compute the SUV matrces
    s, u, v = tf.linalg.svd(...)

    s = tf.linalg.diag(...)

    # ### Exercise 5:
    # 
    # Now compute the PCA for 2 principal components. See how the shape gets modified from original dataset and PCA dataset.

    k = 2
    pca = tf.matmul(...)

    print('original data shape',train.shape)
    print('reduced data shape', pca.shape)

    # ### Exercise 6:
    # Let us plot and see if our PCA is able to cluster the dataset.

    plt.scatter(...)

    # ## Optional Exercise
    # 
    # Repeating the clustering process this time using the K-means algorithm on the Iris dataset. Reflect on the result.

if __name__ == "__main__":
    main()




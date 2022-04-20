
# # Autoencoders
# Autoencoders are a class of neural network that attempt to recreate the input
# as their target using back-propagation. An autoencoder consists of two parts; an **encoder** and a **decoder**. The encoder will read the input and compress it to a compact representation, and the decoder will read the compact representation and recreate the input from it. In other words, the autoencoder tries to learn the identity function by minimizing the reconstruction error. They have an inherent capability to learn
# a compact representation of data. They are at the center of deep belief networks
# and find applications in image reconstruction, clustering, machine translation,
# and much more.
# 
# This exercise aims to test your understanding of autoencoder architecture, and how it can be used to denoise an image. We will build a convolutional autoencoder. Combining your knowledge of a Vanilla/Denoising Autoencoder and Convolutional Networks.
# 
# The notebook has five Exercises followed by an optional exercise.

#@title Import Modules 
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow.keras as K
import tensorflow as tf
from auto import Autoencoder
from helper import Helper

def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    # ## Denoising Autoencoder
    # 
    # When we train the autoencoder, we can train it directly on the raw images or we can add noise to the input images while training. When the autoencoder is trained on noisy data, it gets an even interesting property--it can reconstruct noisy images. In other words--you give it an image with noise and it will remove the noise from it.

    # ## Exercise 1:
    # In this exercise we will train the stacked autoencoder in four steps:
    # * In [Step 1](#step1) choose the noise = 0
    # * Complete the [Step 2](#step2)
    # * In the [Step 3](#step3) choose filters as [16, 32, 64] for Encoder and [64, 32, 16] for Decoder.
    # * Perform [Step 4](#step4) for batch size of 64 and 10 epochs
    # * Reflect on the plotted images what do you see?

    # **Answer 1** (Double click to edit)*italicized text*

    # <a id='step1'></a>
    # ### Step 1:
    # Read the dataset, process it for noise = 0

    #@title Dataset Reading and Processing
    Noise = 0.8 #@param {type:"slider", min:0, max:1, step:0.1}
    (x_train, _), (x_test, _) = K.datasets.mnist.load_data()

    x_train = x_train / 255.
    x_test = x_test / 255.

    x_train = np.reshape(x_train, (len(x_train),28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    noise = Noise
    x_train_noisy = x_train + noise * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0, 1)
    x_test_noisy = np.clip(x_test_noisy, 0, 1)

    x_train_noisy = x_train_noisy.astype('float32')
    x_test_noisy = x_test_noisy.astype('float32')

    # ## Exercise 2:
    # In this exercise we will make only one change, in step 3 choose filters as: `[16, 32, 64]` for both Encoder and Decoder.
    #  Try training the Autoencoder. What happens? Why do you think it is so?

    # **Answer 2** (Double click to edit)

    # ## Exercise 3:
    # 
    # Now we will introduce noise of 0.2 in the training dataset. Train an autoencoder with filters [64,32,16] for encoder and [16,32,64] for decoder and observe the reconstrucred images.
    # 
    # 
    # What do you find? Is the autoencoder able to recognize noisy digits?
    # 

    # **Answer 3** (Double click to edit)

    # ## Exercise 4:
    # 
    # Let us be more adventurous with the same Encoder-Decoder architecture, we increase the noise and observe the reconstrucred images.
    # 
    # 
    # What do you find? Till what noise value is the autoencoder able to reconstruct images? Till what noise level you (human) can recognize the digits in the noisy image.
    # 

    # **Answer 4** (Double click to edit)

    # <a id='step3'></a>

    # ### Step 3:
    # 
    # We have built Convolutional Autoencoder. That is both Encoder and Decoder are buit using Convolutional layers. Below you need to select 

    #@title Select Filters for Encoder & Decoder
    filter_encoder_0 = 64 #@param {type:"slider", min:8, max:256, step:2}
    filter_encoder_1 = 32 #@param {type:"slider", min:8, max:256, step:2}
    filter_encoder_2 = 16 #@param {type:"slider", min:8, max:256, step:2}

    filters_en = [filter_encoder_0,filter_encoder_1,filter_encoder_2]


    filter_decoder_0 = 16 #@param {type:"slider", min:8, max:256, step:2}
    filter_decoder_1 = 32 #@param {type:"slider", min:8, max:256, step:2}
    filter_decoder_2 = 64 #@param {type:"slider", min:8, max:256, step:2}

    filters_de = [filter_decoder_0,filter_decoder_1,filter_decoder_2]

    model = Autoencoder(filters_en, filters_de)

    model.compile(loss='binary_crossentropy', optimizer='adam')


    # ### Step 4:
    # Choose the appropriate batch_size and epochs

    #@title Train the model
    BATCH_SIZE = 64 #@param {type:"slider", min:32, max:2000, step:10}
    EPOCHS = 10 #@param {type:"slider", min:1, max:100, step:1}
    batch_size = BATCH_SIZE
    max_epochs = EPOCHS
    checkpoint_path = "../../Checkpoints/auto_model.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    Helper.printline("Start training")
    loss = model.fit(x_train_noisy,
                    x_train,
                    validation_data=(x_test_noisy, x_test),
                    epochs=max_epochs,
                    batch_size=batch_size,
                    callbacks=[cp_callback])  # Pass callback to training
    Helper.printline("End training")
    Helper.printline(f"x_test_noisy: {x_test_noisy}")
    #@title Reconstructed images
    number = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    Helper.printline("START: Get test model")
    test_model = model(x_test_noisy)
    Helper.printline("END: Get test model")
    for index in range(number):
        print(f"index = {index}")
        # display original
        ax = plt.subplot(2, number, index + 1)
        plt.imshow(x_test_noisy[index].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, number, index + 1 + number)
        item = tf.reshape(test_model[index], (28, 28))
        plt.imshow(item, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    Helper.printline("---------- Save print image ------------")
    plt.savefig("../../Images/autoencoder_graph.png")
    Helper.printline("---------- End print image ------------")
    # ## Optional Exercise
    # Construct a Sparse Autoencoder with Dense layer/s, train it on noisy images as before. See how the hidden dimensions influence the reconstruction. Which is one is better for denoising, the convolution Encoder/Decoder or Dense Encoder/Decoder, why?

def print_results():
    Noise = 0
    (x_train, _), (x_test, _) = K.datasets.mnist.load_data()
    x_test = x_test / 255.
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    noise = Noise
    x_test_noisy = x_test + noise * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_test_noisy = np.clip(x_test_noisy, 0, 1)
    x_test_noisy = x_test_noisy.astype('float32')
    checkpoint_path = "../../Checkpoints/auto_model.ckpt"
    filter_encoder_0 = 16 #@param {type:"slider", min:8, max:256, step:2}
    filter_encoder_1 = 32 #@param {type:"slider", min:8, max:256, step:2}
    filter_encoder_2 = 64 #@param {type:"slider", min:8, max:256, step:2}

    filters_en = [filter_encoder_0,filter_encoder_1,filter_encoder_2]


    filter_decoder_0 = 64 #@param {type:"slider", min:8, max:256, step:2}
    filter_decoder_1 = 32 #@param {type:"slider", min:8, max:256, step:2}
    filter_decoder_2 = 16 #@param {type:"slider", min:8, max:256, step:2}

    filters_de = [filter_decoder_0,filter_decoder_1,filter_decoder_2]

    model = Autoencoder(filters_en, filters_de)

    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.load_weights(checkpoint_path)
    number = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    Helper.printline("START: Get test model")
    test_model = model(x_test_noisy)
    Helper.printline("END: Get test model")
    for index in range(number):
        print(f"index = {index}")
        # display original
        ax = plt.subplot(2, number, index + 1)
        plt.imshow(x_test_noisy[index].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, number, index + 1 + number)
        item = tf.reshape(test_model[index], (28, 28))
        plt.imshow(item, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    Helper.printline("---------- Save print image ------------")
    plt.savefig("../../Images/autoencoder_graph.png")
    Helper.printline("---------- End print image ------------")
    
if __name__ == "__main__":
    main()
    #print_results()


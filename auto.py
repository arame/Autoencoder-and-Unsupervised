import tensorflow.keras as K
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D

# ## AutoEncoder  Architecture
# The number of hidden units in the autoencoder is typically less than the number of input (and output) units. This forces the encoder to learn a compressed representation of the input, which the decoder reconstructs. If there is a structure in the input data in the form of correlations between input features, then the autoencoder will discover some of these correlations, and end up learning a low-dimensional representation of the data similar to that learned using principal component analysis (PCA).
# 
# Once trained
# * We can discard **decoder** and use **Encoder** to optain a compact representation of input.
# * We can cascade Encoder to a classifier.
# 
# The encoder and decoder components of an autoencoder can be implemented using either dense, convolutional, or recurrent networks, depending on the kind of data that is being modeled.
# 
# Below we define an encoder and a decoder using Convolutional layers. Both consist of three convolutional layers. Each layer in Encoder has a corresponding layer in decoder, thus in this case it is like three autoencoders stacked over each other. This is also called **Stacked Autoencoders** 
# 
# ![](https://drive.google.com/uc?id=1UzM67qf1VE_8akrCgiohKjUIHoO_2x4E)
# 
# 
    
#@title Encoder
class Encoder(K.layers.Layer):
    def __init__(self, filters):
        super(Encoder, self).__init__()
        self.conv1 = Conv2D(filters=filters[0], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv2 = Conv2D(filters=filters[1], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv3 = Conv2D(filters=filters[2], kernel_size=3, strides=1, activation='relu', padding='same')
        self.pool = MaxPooling2D((2, 2), padding='same')
            
    
    def call(self, input_features):
        #print("Encoder call")
        x = self.conv1(input_features)
        #print("Ex1", x.shape)
        x = self.pool(x)
        #print("Ex2", x.shape)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x
        

#@title Decoder
class Decoder(K.layers.Layer):
    def __init__(self, filters):
        super(Decoder, self).__init__()
        self.conv1 = Conv2D(filters=filters[2], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv2 = Conv2D(filters=filters[1], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv3 = Conv2D(filters=filters[0], kernel_size=3, strides=1, activation='relu', padding='valid')
        self.conv4 = Conv2D(1, 3, 1, activation='sigmoid', padding='same')
        self.upsample = UpSampling2D((2, 2))

    def call(self, encoded):
        x = self.conv1(encoded)
        #print("dx1", x.shape)
        x = self.upsample(x)
        #print("dx2", x.shape)
        x = self.conv2(x)
        x = self.upsample(x)
        x = self.conv3(x)
        x = self.upsample(x)
        return self.conv4(x)

# <a id='another_cell'></a>
# ### Step 2
# 
# You need to complete the code below. We will be using the Encoder and Decoder architectures that we have defined above to build an autoencoder. In the code below replace `...` with right code. 

class Autoencoder(K.Model):
    def __init__(self, filters_encoder, filters_decoder):
        super(Autoencoder, self).__init__()
        self.loss = []
        self.encoder = Encoder(filters_encoder)
        self.decoder = Decoder(filters_decoder)

    def call(self, input_features):
        #print(f"Input features shape: {input_features.shape}")
        encoded = self.encoder(input_features)
        #print(f"Input features shape: {encoded.shape}")
        reconstructed = self.decoder(encoded)
        #print(f"Reconstructed shape {reconstructed.shape}")
        return reconstructed
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

from GAN_for_STAR import make_generator_model, make_discriminator_model


if __name__ == '__main__':

    generator = make_generator_model()
    discriminator = make_discriminator_model()
    #generator.load_weights(tf.train.latest_checkpoint('./training_checkpoints'))
    generator.load_weights(tf.train.latest_checkpoint('./weights'))
    #generator.load_weights('./weights/weights_3_layers_gen.data-00000-of-00001')



    data = loadmat('projs_for_training_500.mat')

    examples = data['PROJ_RF'][0:10,:,:,:]
    gts = data['PROJ_IM'][0:10,:,:]

    outputs = generator.predict(examples)

    judgements = discriminator.predict(outputs)
    print(discriminator.output_shape)
    print("Judgements: ", judgements)

    plt.figure(1)
    plt.gray()

    for u in range(5):
        plt.subplot(2,5,u+1)
        plt.imshow(gts[u+5,:,:].astype(float))
        plt.subplot(2,5,u+6)
        plt.imshow(outputs[u+5,:,:,0].astype(float))


    plt.show()
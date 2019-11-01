from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat


###########################################

###########################################





def make_generator_model():
    # This generates a 32 x 51 output from a 32 x 51 x 2 input
    # input - (32, 51, 2)
    model = tf.keras.Sequential()

    model.add(layers.Flatten(input_shape=(32, 51, 2)))
    model.add(layers.Dense(32 * 51 * 4, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    num_layers = 5
    for u in range(num_layers-1):
        model.add(layers.Flatten())
        model.add(layers.Dense(32*51*4,use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

    model.add(layers.Reshape((32,51,4)))
    assert model.output_shape == (None, 32, 51, 4)

    #
    model.add(layers.Conv2DTranspose(64,(5,5), strides=(1,1), padding='same',
                                     use_bias=False)) #input_shape=(32,51,2))) # a real channel and an imaginary channel
    # Deleted densely connected layer
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None,32,51,64)

    model.add(layers.Conv2DTranspose(32, (5,5), strides=(1,1), padding='same',use_bias=False))
    assert model.output_shape == (None,32,51,32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5,5), strides=(1,1),padding='same',use_bias=False))#, activation='tanh'))
    assert model.output_shape == (None,32,51,1)

    print('Generator model: ')
    model.summary()

    return model


def make_discriminator_model():
    # Input - (32, 51, 1)
    # Output - single number
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5,5), strides=(1,1), padding='same',
                            input_shape=[32, 51, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5,5), strides=(1,1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1,activation='relu'))



    print('Discriminator model: ')
    model.summary()

    return model


# TODO
def discriminator_loss(real_output, model_output): # TODO Wasserstein
    real_loss = cross_entropy(tf.ones_like(real_output),real_output)
    model_loss = cross_entropy(tf.zeros_like(model_output),model_output)

    #self.discriminator_loss = tf.reduce_mean(logits_real - logits_fake)

    return real_loss + model_loss
def generator_loss(real_output, model_output): # TODO Wasserstein
    eta = 0.5
    model_loss = cross_entropy(tf.ones_like(model_output),model_output)
    L2_loss = mean_squared_error(real_output, model_output)

    #self.gen_loss = tf.reduce_mean(logits_fake)

    return eta*model_loss + (1-eta)*L2_loss


@tf.function # Should be fine ..?
def train_step(im_proj, rf_proj):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(rf_proj, training=True)
        real_output = discriminator(im_proj, training=True)
        model_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(real_output, model_output)
        disc_loss = discriminator_loss(real_output, model_output)



    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))# take a step (gen)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables)) # Take a step (disc)

    return gen_loss, disc_loss, generated_images




def train(dataset, epochs):

    gen_loss_all = np.zeros((epochs, BATCH_SIZE))
    disc_loss_all = np.zeros((epochs, BATCH_SIZE))

    for epoch in range(epochs):
        print('Starting another epoch')
        start = time.time()

        for batch_ind, data_batch in enumerate(dataset):
            proj_im_batch = np.expand_dims(data_batch[:,:,:,0], axis=3)
            proj_rf_batch = data_batch[:,:,:,1:]
            gen_loss, disc_loss, __ = train_step(proj_im_batch, proj_rf_batch)
            gen_loss_all[epoch,batch_ind] = gen_loss.numpy()
            disc_loss_all[epoch,batch_ind] = disc_loss.numpy()

            print('Epoch {:d} Batch {:d} Generator loss: '.format(epoch, batch_ind), gen_loss.numpy())
            print('                      Discriminator loss: ', disc_loss.numpy())

        if (epoch + 1) % 5 == 0:
            #checkpoint.save(file_prefix = checkpoint_prefix)
            generator.save_weights('./weights/gen_weights')
            #discriminator.save_weights('./weights/weights_4_layers_disc')


        print('Time for epoch {} is {} sec'.format(epoch+1,time.time()-start))

    return gen_loss_all, disc_loss_all


if __name__ == '__main__':
    ### Set up
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mean_squared_error = tf.keras.losses.MeanSquaredError()

    # Make models
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    # Initial test of models
    #
    # Optimizers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)  # 1e-4 is learning rate
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)  #

    # Save checkpoints
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    """
    TRAINING COMMENCES BELOW
    """
    # Load dataset and train
    BUFFER_SIZE = 500
    BATCH_SIZE = 50
    EPOCHS = 15
    train_data = loadmat('projs_for_training_500.mat')

    train_data['PROJ_IM'] = train_data['PROJ_IM'].astype('float32')
    train_data['PROJ_RF'] = train_data['PROJ_RF'].astype('float32')

    print(np.shape(train_data['PROJ_IM']))
    print(np.shape(train_data['PROJ_RF']))

    train_data_array = np.zeros((500,32,51,3))
    train_data_array[:,:,:,0] = train_data['PROJ_IM']

    # Normalize im projs
    for u in range(500):
        this_proj_im = train_data_array[u,:,:,0]
        normalized_proj_im = (this_proj_im - this_proj_im.mean()) / this_proj_im.std()
        train_data_array[u,:,:,0] = normalized_proj_im




   # train_data_array[:,:,:,1:] = train_data['PROJ_RF']

    train_data_array[:,:,:,1:] = loadmat('projs_rf_mag_phase_500.mat')['PROJ_RF']


    train_dataset = tf.data.Dataset.from_tensor_slices(train_data_array).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    # Batch it while keeping the pairing!

    gen_loss_all, disc_loss_all = train(train_dataset, EPOCHS)

    # Save loss
    savemat('./losses_5_layers_4x_notanh_e15_mp.mat',{'gen_losses': gen_loss_all, 'disc_losses': disc_loss_all})

    # Save weights


   # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    ##
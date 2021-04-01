from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import tensorflow as tf
from matplotlib import pyplot as plt


import os
from glob import glob
import math
import numpy as np
from util import *


class GAN():
    def __init__(self):
        self.img_rows = 218
        self.img_cols = 173
        self.channels = 3
        self.input_shape = self.img_rows * self.img_cols
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        optimizer = Adam(0.0002)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer ,metrics=['accuracy'])
        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.input_shape,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # General process
        # input: noise => generates images by generator => generated images is validated by the discriminator.
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        noise_shape = (self.input_shape,)

        model = Sequential()

        model.add(Dense(100, input_shape=noise_shape))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))

        model.add(Dense(200))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))

        model.add(Dense(100))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))

        model.add(Dense(self.input_shape * self.channels, activation = "sigmoid"))
        model.add(Reshape([self.img_rows, self.img_cols, self.channels]))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)
        return Model(noise, img)

    def build_discriminator(self):
        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()
        model.add(Flatten(input_shape=img_shape))

        model.add(Dense(100))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))

        model.add(Dense(200))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))

        model.add(Dense(100))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))
        
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)
        return Model(img, validity)

    def add_noise(self ,image):
        ch = self.channels
        row ,col = self.img_rows, self.img_cols
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        image = tf.image.resize(noisy, (row, col))
        return image

    def train(self, epochs, batch_size=128, save_interval=50):
        # Please change to where you dataset is!
        data_dir = "img_align_celeba"
        filepaths = os.listdir(data_dir)
        half_batch = int(batch_size / 2)
        # Lists to log the losses of discriminator real, discriminator fake, and generator.
        d_loss_logs_r = []
        d_loss_logs_f = []
        g_loss_logs = []
        n_iterations = math.floor(len(filepaths ) /batch_size)
        print(n_iterations)
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            for ite in range(n_iterations):
                # Select a random half batch of images
                X_train = get_batch(glob(os.path.join(data_dir, '*.jpg'))[ite*batch_size:(ite +1 ) *batch_size], self.img_cols, self.img_rows, 'RGB')
                # Normalize train_data to be between -1 and 1 (similar to /255)
                X_train = (X_train.astype(np.float32) - 127.5) / 127.5
                X_train =np.array([self.add_noise(image) for image in X_train])
                idx = np.random.randint(0, X_train.shape[0], half_batch)
                imgs = X_train[idx]
                noise = np.random.normal(0, 1, (half_batch, self.input_shape))
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------
                noise = np.random.normal(0, 1, (batch_size, self.input_shape))
                # The generator wants the discriminator to label the generated samples
                # as valid (ones)
                valid_y = np.array([1] * batch_size)
                # Train the generator
                g_loss = self.combined.train_on_batch(noise, valid_y)
                # Plot the progress
                print("%d %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, ite, d_loss[0], 100 * d_loss[1], g_loss))

                # Append the logs with the loss values in each training step
                d_loss_logs_r.append([epoch, d_loss[0]])
                d_loss_logs_f.append([epoch, d_loss[1]])
                g_loss_logs.append([epoch, g_loss])

                d_loss_logs_r_a = np.array(d_loss_logs_r)
                d_loss_logs_f_a = np.array(d_loss_logs_f)
                g_loss_logs_a = np.array(g_loss_logs)

                # If the iteration is at the save intervals, save the generated images.
                # A way to make sure things are running fine
                if ite % save_interval == 0:
                    save_imgs(self.generator, self.img_row, self.img_col, epoch,ite)
                    plt.plot(d_loss_logs_r_a[:, 0], d_loss_logs_r_a[:, 1], label="Discriminator Loss - Real")
                    plt.plot(d_loss_logs_f_a[:, 0], d_loss_logs_f_a[:, 1], label="Discriminator Loss - Fake")
                    plt.plot(g_loss_logs_a[:, 0], g_loss_logs_a[:, 1], label="Generator Loss")
                    plt.xlabel('Epochs-iterations')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.title('Variation of losses over epochs')
                    plt.grid(True)
                    plt.show()

    def saveGeneratorWeights(self, epoch):
        self.generator.save_weights("model" + str(epoch) + ".h5")
        print("Saved model to disk")

    def saveGeneratorInJson(self, epoch):
        model_json = self.generator.to_json()
        with open("model" + str(epoch) + ".json", "w") as json_file:
            json_file.write(model_json)

    def saveGenerator(self):
        tf.saved_model.save(self.generator, 'homework3/celebA')
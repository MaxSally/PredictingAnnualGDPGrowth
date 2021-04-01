from __future__ import print_function, division

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from model import *

def get_image(image_path, width, height, mode):
    image = Image.open(image_path)
    image = image.resize([width, height])
    return np.array(image.convert(mode))

def get_batch(image_files, width, height, mode):
    #print(image_files)
    data_batch = np.array([get_image(sample_file, width, height, mode) for sample_file in image_files])
    return data_batch

def save_imgs(generator, row, col, epoch,iteration):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, row * col))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = (1/2.5) * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(str(epoch)+"-"+str(iteration)+".png")
    plt.close()

def plot(d_loss_logs_r_a,d_loss_logs_f_a,g_loss_logs_a):
    #Generate the plot at the end of training
    #Convert the log lists to numpy arrays
    d_loss_logs_r_a = np.array(d_loss_logs_r_a)
    d_loss_logs_f_a = np.array(d_loss_logs_f_a)
    g_loss_logs_a = np.array(g_loss_logs_a)
    plt.plot(d_loss_logs_r_a[:,0], d_loss_logs_r_a[:,1], label="Discriminator Loss - Real")
    plt.plot(d_loss_logs_f_a[:,0], d_loss_logs_f_a[:,1], label="Discriminator Loss - Fake")
    plt.plot(g_loss_logs_a[:,0], g_loss_logs_a[:,1], label="Generator Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Variation of losses over epochs')
    plt.grid(True)
    plt.show()
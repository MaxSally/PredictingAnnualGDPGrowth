from __future__ import print_function, division

from model import *
from util import *


gan = GAN()
gan.train(epochs=6, batch_size=50, save_interval=200)

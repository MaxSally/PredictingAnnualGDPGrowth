from __future__ import print_function, division

from model import *
from util import *


gan = GAN()
gan.train(epochs=20, batch_size=512, save_interval=50)

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

#losses = loadmat('./losses_from_big/losses_3.mat')
losses = loadmat('losses_5_layers_4x_Rectified.mat')
print(losses.keys())
gl = losses['gen_losses'].mean(axis=1)
dl = losses['disc_losses'].mean(axis=1)

plt.figure(1)
plt.subplot(211)
plt.plot(gl)
plt.title('Generator loss over training steps')
plt.xlabel('Step #')

plt.figure(1)
plt.subplot(212)
plt.plot(dl)
plt.title('Discriminator loss over training steps')
plt.xlabel('Step #')

plt.show()
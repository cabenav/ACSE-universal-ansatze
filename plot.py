import numpy as np
import matplotlib.pyplot as plt

######################### config #######################3
result_folder='checkpoints'
filename='m4-float32-batchsize2048-layers16_512_512_512_512_512_10-loss.pt'
#filename='m4-float32-batchsize512-layers16_512_512_512_512_512_10-loss.pt'
#filename='m4-float32-batchsize1024-layers16_512_512_512_512_512_10-loss.pt'
filename_loss=f'{result_folder}/{filename}'

import sys
#print ('argument list', sys.argv)
try:
    filename_loss = sys.argv[1]
    print('get filename_loss:',filename_loss)
    filename=filename_loss.split('/')[-1]    
except:
    print('no input loss file from cmd')

fig_folder='fig'
filename_fig =f'{fig_folder}/{filename}.pdf'
    
# print local variables
local=locals().copy()
for k in local:
    if k[0:2] != '__':      #skip built in module
        print(f'{k}:\t{local[k]}')


import torch
loss_np_array = torch.load(filename_loss)
data=loss_np_array
print('sample of 10 data points')
print(data[:10])
print(data)

scale=1
while len(data) > 2000:
    data = data[::10]
    scale = scale * 10

# increase sigma you can get a more smoothed function.
from scipy.ndimage.filters import gaussian_filter1d
data = gaussian_filter1d(data, sigma=2)

plt.figure()
plt.plot(data, label='loss')
plt.ylim(0.001, 0.5)

plt.title(filename)
plt.ylabel("Loss /log")
plt.xlabel(f"Epoches x {scale}")
plt.yscale('log')
plt.legend(loc=0);
plt.savefig(filename_fig)

print(f'{filename_loss} has been ploted and saved into {filename_fig}')

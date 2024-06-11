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
print('data.shape',data.shape)
print('sample of data points: first 10 and last 10')
print(data[:5])
print(data[-5:])
#print(data)

scale=1
if False and len(data) > 2000:
    data = data[::10]
    scale = scale * 10
    print('sparsify data to ',data.shape)

# increase sigma you can get a more smoothed function.
#from scipy.ndimage import gaussian_filter1d
#data = gaussian_filter1d(data, sigma=2)

plt.figure(figsize=(9, 7))
#figsize=(9, 11)
print(data.shape)

labels= ['validation loss','training loss','err','best err', 'epoch']

if data.shape[1] == 2:
    plt.plot(data, label=['validation loss','training loss'])
elif data.shape[1] == 5:
    #print(data)
    if True:
        data[:,2:4] = data[:,2:4]/100
        labels= ['validation loss','training loss','err/100','best err/100', 'epoch']
    plt.plot(data[:,:4], label=labels[:4])
    #plt.plot(data[:,:4], label=['validation loss','training loss','acc','best acc'])
else:
    plt.plot(data, label='validation loss')
#plt.ylim(0.001, 0.9)
#plt.tight_layout()
plt.title(filename)
plt.ylabel("Loss /log")
#plt.xlabel(f"Epoches x {scale}")
plt.xlabel(f"Epoches")
plt.yscale('log')
plt.legend(loc=0);
plt.savefig(filename_fig)

print(f'{filename_loss} has been ploted and saved into {filename_fig}')

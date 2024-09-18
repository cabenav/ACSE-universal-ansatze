import numpy as np
import matplotlib.pyplot as plt
import json
import os
######################### config #######################3
result_folder='checkpoints'
filename='m4-float32-batchsize2048-layers16_512_512_512_512_512_10-loss.pt'
#filename='m4-float32-batchsize512-layers16_512_512_512_512_512_10-loss.pt'
#filename='m4-float32-batchsize1024-layers16_512_512_512_512_512_10-loss.pt'
filename_loss=f'{result_folder}/{filename}'
fig_folder='fig'



import sys
#print ('argument list', sys.argv)
try:
    filename_loss = sys.argv[1]
    print('get filename_loss:',filename_loss)
    filename=filename_loss.split('/')[-1]    
except:
    print('no input loss file from cmd')


filename_fig =f'{fig_folder}/{filename}.pdf'
filename_config_json = filename_loss[:-8] +'.json'

if os.path.exists(filename_config_json):

    with open(filename_config_json,'r') as f:
        cfg = json.load(f)
        try:
            title = f"file: {filename}\nlayers: {cfg['LAYERS']}\nbatch_size={cfg['batch_size']}, learning rate={cfg['learning_rate']}, gpu={cfg['gpu']}"
        except:
            title = f"file: {filename}\nbatch_size={cfg['batch_size']}, learning rate={cfg['learning_rate']}, gpu={cfg['gpu']}"
else:
    title=filename

# print local variables
local=locals().copy()
for k in local:
    if k[0:2] != '__':      #skip built-in modules
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
        #data[:,2:4] = data[:,2:4]/10
        labels= ['validation loss on Ansatz parameter','training loss on Ansatz paramter','MSE error on energy.min()','best error', 'epoch']
    plt.plot(data[:,:4], label=labels[:4])
    for i in [0,3]:
        # display value of the last point
        x=data.shape[0]
        y=data[-1,i]
        #print( filename_fig,'xy',x,y)
        plt.annotate(f'({y:.2f})', (x, y), ha='center',va='top')
    #plt.plot(data[:,:4], label=['validation loss','training loss','acc','best acc'])
else:
    plt.plot(data, label='validation loss')
#plt.ylim(0.001, 0.9)
#plt.tight_layout()
#plt.title(filename)
plt.title(title,loc='left')
plt.ylabel("Loss /log")
#plt.xlabel(f"Epoches x {scale}")
plt.xlabel(f"Epoches")
plt.yscale('log')
plt.legend(loc=0);
plt.savefig(filename_fig)

print(f'{filename_loss} has been ploted and saved into {filename_fig}')

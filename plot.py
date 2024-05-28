import numpy as np
import matplotlib.pyplot as plt


# config
result_folder='checkpoints'
filename='m4-float32-batchsize2048-layers16_512_512_512_512_512_10-loss.pt'
filename='m4-float32-batchsize512-layers16_512_512_512_512_512_10-loss.pt'
#filename='m4-float32-batchsize1024-layers16_512_512_512_512_512_10-loss.pt'
filename_loss=f'{result_folder}/{filename}'
fig_folder='fig'
filename_fig =f'{fig_folder}/{filename}.pdf'

import sys
#print ('argument list', sys.argv)
try:
    filename_loss = sys.argv[1]
    print('get filename_loss:',filename_loss)
except:
    print('no imput loss file from cmd')

# print local variables
local=locals().copy()
for k in local:
    if k[0:2] != '__':
        print(f'{k}:\t{local[k]}')

#print('input/output files:',filename_prefix,filename_checkpoint,filename_loss)
exit()

#folder='checkpoints'
#filename_loss = '{folder}loss.pt'
#filename_fig = ''
#print('input/output files:',filename,filename_checkpoint,filename_loss)

import torch
#loss_list
loss_np_array = torch.load(filename_loss)
#print(loss_list[:10])
#a = torch.tensor(loss_list)
#print(a[:10])


data=loss_np_array
print('sample of 10 data points')
print(data[:10])
plt.figure()
plt.plot(data, label='loss')

plt.title(filename)
plt.ylabel("Loss /log")
plt.xlabel("Epoches")
plt.yscale('log')
plt.legend(loc=0);
plt.savefig(filename_fig)

print(f'{filename_loss} has been ploted and saved into {filename_fig}')

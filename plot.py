import numpy as np
import matplotlib.pyplot as plt


#config
L=5
folder='../../DMRG/tenpy/data'
filename=f'{folder}/data-ising-L{L}-1.pt'  # 41450 entries
filename=f'{folder}/data-ising-L{L}-2.pt'  # 84950 entries
filename=f'{folder}/data-ising-L5.dict.pt.array'
_=filename.split('/')[-1]

filename_checkpoint=f'result/{_}.32'
filename_loss=f'result/{_}.loss.32'
filename_loss = 'loss.pt'

print('input/output files:',filename,filename_checkpoint,filename_loss)

import torch

loss_list = torch.load(filename_loss)
print(loss_list[:10])
a = torch.tensor(loss_list)
print(a[:10])

#data = np.array(loss_list)
#data = loss_list.cpu()
#print(data)

data=a
plt.figure()
plt.plot(data)

plt.title(filename_loss)
plt.ylabel("Loss /log")
plt.xlabel("Epoches")
plt.yscale('log')
#plt.legend(loc=0);
plt.savefig('fig/loss.pdf')

def plot():
    torch.load(filename_loss)
    

if __name__=="__main__":
    #plot()
    print('done')

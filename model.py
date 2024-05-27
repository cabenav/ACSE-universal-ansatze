
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm
import torch

######################### config start ###############################
folder='data'
#filename=f'{folder}/data-ising-L5.dict.pt.array'
#_=filename.split('/')[-1]
#filename_checkpoint=f'result/{_}.32'
#filename_loss=f'result/{_}.loss.32'
folder='checkpoints'
filename_checkpoint=f'{folder}/check.pt'
filename_loss=f'{folder}/loss.pt'

#filename='entry50000.npy'
#filename='entry321200.npy'
filename='entry2295600.npy'
filename_prefix='data/m4'
print('input/output files:',filename,filename_checkpoint,filename_loss,filename_prefix)

output_width=10
hidden_size= 64*8
num_hidden_layers=5
LAYERS= [hidden_size for _ in range(num_hidden_layers+2)]
LAYERS[0]=16
LAYERS[-1]=output_width
n_epochs = 25000 #250   # number of epochs to run
batch_size = 64*4 * 8 #10  # size of each batch
#torch.set_printoptions(8)
torch.set_printoptions(linewidth=140)
#torch.set_default_dtype(torch.float64)
######################### config end   ###############################

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# to choose gpu
# CUDA_VISIBLE_DEVICES=1,2 python myscript.py
print(f"Using {device} device")

import os
import glob
def load(filename_prefix): #loadd all files with this filename prefix
    filelist = glob.glob(f'{filename_prefix}*.npy')
    print(f'loading {len(filelist)} data files')
    print('get file list',filelist)
    data_list=[]
    for filename in filelist:
        data_list.append(np.load(filename))
    data = np.concatenate( data_list)
    return data


# load training data
print(f'loading data: {filename_prefix}')
#d = torch.load(filename)
#d = np.load(filename)
d = load('data/m4')
d=torch.tensor(d,device=device)
print('sample entry d[0]')
print(d[0])
d=d.float()  #differ by 1e-9
X = d[:,:16]
y = d[:,-10:]
print('data shape X Y',X.shape,y.shape)
print(type(X),X.dtype)

#for evaluation
X_test,y_test = X[:1000],y[:1000]
print('test shape X Y',X_test.shape,y_test.shape)

#exit()

class Deep(nn.Module):
    def __init__(self,layers=[28*28,640,640,60,10]):
        super().__init__()
        modules=[]
        print('processing layers:',layers)
        num_layers=len(layers)
        for i in range(num_layers-2):
            layer0 = layers[i]
            layer1 = layers[i+1]
            layer = nn.Linear(layer0,layer1)
            act = nn.ReLU()
            modules.append(layer)
            modules.append(act)
        self.linear_relu_stack = nn.Sequential(*modules)
        self.output = nn.Linear(layers[-2], layers[-1])
        
    def forward(self, x):
        x = self.linear_relu_stack(x)
        x = self.output(x)
        return x


def model_train(model, X_train, y_train, X_val, y_val,best_acc=-np.inf,best_weights = None):
    for i in [X_train, y_train, X_val, y_val]:
        print(i.shape)
    # loss function and optimizer
    loss_fn = nn.MSELoss()
    loss_list=[]
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    batch_start = torch.arange(0, len(X_train), batch_size)

    for epoch in range(n_epochs):
        model.train()
        # permutate input data order randomly
        indices = torch.randperm(X_train.size()[0])
        X_train=X_train[indices]
        y_train=y_train[indices]
        X_test,y_test = X_train[-1000:],y_train[-1000:]
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
            bar.set_description(f"Epoch {epoch}/{n_epochs}")            
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                #X_batch,y_batch = X_batch.to(device),y_batch.to(device)
                #print('X_batch.shape',X_batch.shape)
                
                # forward pass
                y_pred = model(X_batch)
                #print(y_pred.shape,y_batch.shape)
                loss = loss_fn(y_pred, y_batch)
                
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress                
                #acc = acc_eval(y_pred,y_batch)
                acc = - loss
                #print(acc)
                bar.set_postfix(
                    loss=float(loss),
                    best_acc = float(best_acc),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        #X_val=X_val.to(device)
        #y_val=y_val.to(device)        
        y_pred = model(X_val)
        #acc = ((y_pred>0) == y_val).type(torch.float).mean()
        #acc = acc_eval(y_pred,y_val)
        loss = loss_fn(y_pred,y_val)
        loss_list.append( loss.detach().cpu().item() )
        #loss_list.append( loss.numpy() )
        loss_np_array = np.array(loss_list)
        #print(loss_list)
        #print(loss_np_array)
        torch.save(loss_np_array,filename_loss)
        #torch.save(loss_list,filename_loss)
        print(f'loss list saved into {filename_loss}')
        acc = - loss
        #print( ((y_pred-y_val)/y_val).abs() )
        
        print(y_pred)
        print(y_pred-y_val)
        print(y_val)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
            #save into file
            print('saving data')
            torch.save(best_weights,filename_checkpoint)
            print(f'weights saved into {filename_checkpoint} at epoch={epoch}, acc={acc}')
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc,best_weights


#cv_scores = []

# train the same model the the same data a few times
layers=LAYERS    
model = Deep(layers).to(device)
print(model)


# Hold the best model
best_acc = - np.inf   # init to negative infinity
best_weights = None

model_train(model, X, y, X_test, y_test, best_acc, best_weights)


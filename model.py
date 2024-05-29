
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm
import torch

######################### config start ###############################
hidden_size= 64 * 8 * 4
num_hidden_layers=5
LAYERS= [hidden_size for _ in range(num_hidden_layers+2)]
LAYERS[0]=16
LAYERS[-1]=6
n_epochs = 25000 #250   # number of epochs to run
batch_size = 64*4 * 1  #10  # size of each batch
#torch.set_printoptions(8)
torch.set_printoptions(linewidth=140)
#torch.set_default_dtype(torch.float64)

data_folder='data'
title='m4'
filename_prefix=f'{data_folder}/{title}'
result_folder='checkpoints'
note=f'v4-ReLU-Adam0.0001-shuffle-f32-bs{batch_size}-layers{"_".join( str(_) for _ in LAYERS)}'
filename_checkpoint=f'{result_folder}/{title}-{note}-check.pt'
filename_loss=f'{result_folder}/{title}-{note}-loss.pt'
print('title/note:',title,note)
print('input/output files:',filename_prefix,filename_checkpoint,filename_loss)
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
        _data=np.load(filename)
        assert _data.shape[1] == 42
        data_list.append(_data)
        #print(filename)
        #break
    data = np.concatenate(data_list)
    return data


# load training data
print(f'loading data: {filename_prefix}')
d = load(filename_prefix)
d = d[:int(1e6)] # maximum 1 million data
d = torch.tensor(d,device=device)
print('sample entry d[0]')
print(d[0])

#torch.set_default_dtype(torch.float16)
d=d.float()  #differ by 1e-9
#d=d.half()  # half precision for fast training on large nn
X = d[:,:16]
#y = d[:,-10:]  #energy and parameters
y = d[:,-6:]    #parameters

# use the last 1000 for evaluation
eval_size=d.shape[0]//20
X_test,y_test = X[-eval_size:],y[-eval_size:] 
X,y = X[:-eval_size],y[:-eval_size]

print(type(X),X.dtype)
print('data shape X Y',X.shape,y.shape)
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
            #act = nn.ReLU()
            act = nn.LeakyReLU()
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
    loss_list=[]
    

    batch_start = torch.arange(0, len(X_train), batch_size)
    for epoch in range(n_epochs):
        model.train()
        if True: # whether to shuffle the data
            # permutate input data order randomly
            indices = torch.randperm(X_train.size()[0])
            X_train=X_train[indices]
            y_train=y_train[indices]
        training_loss_list=[]
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
                training_loss_list.append( loss.detach().cpu().item() )
                #print(acc)
                bar.set_postfix(loss=float(loss), best_acc = float(best_acc),acc=float(acc))
        print('sample training loss:','\t'.join( [f'{_:.3f}' for _ in training_loss_list[-100:]]))
        training_loss = np.array(training_loss_list).mean()
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        loss = loss_fn(y_pred,y_val)
        _loss = loss.detach().cpu().item() 
        loss_list.append( (_loss,training_loss) )
        loss_np_array = np.array(loss_list)
        print('validation VS training | loss history')
        print(loss_np_array[-10:])
        
        torch.save(loss_np_array,filename_loss)
        print(f'loss list saved into {filename_loss} at loss={loss_np_array[-1]}')
        acc = - loss

        print('target:     ',end='')
        print((y_val)[0])
        print('prediction: ',end='')
        print(y_pred[0])
        print('diff:       ',end='')
        print((y_pred-y_val)[0])

        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
            #save into file
            print('saving data')
            torch.save(best_weights,filename_checkpoint)
            print(f'best weights saved into {filename_checkpoint} at epoch={epoch}, acc={acc}')            
    model.load_state_dict(best_weights)   # restore model and return best accuracy
    return best_acc,best_weights


model = Deep(LAYERS).to(device)
print(model)
print(f'batch_size={batch_size}')

# Hold the best model
best_acc = - np.inf   # init to negative infinity
best_weights = None

# loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#optimizer = optim.SGD(model.parameters(), lr=0.001)

# load previous result for continuous training
if True:
    try:
        best_weights = torch.load(filename_checkpoint)
        model.load_state_dict(best_weights)
        
        
        model.eval()
        y_pred = model(X_test)
        loss = loss_fn(y_pred,y_test)
        best_acc = - loss
        print('loaded previous weights with best_acc:',best_acc)
    except:
        print('did not find previous weights:',filename_checkpoint)


model_train(model, X, y, X_test, y_test, best_acc, best_weights)


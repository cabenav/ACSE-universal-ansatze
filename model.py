
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm
import torch
from parallel import Parallel

######################### config start ###############################
hidden_size= 64 * 1 * 1
num_hidden_layers=8 # 5
n_epochs = 1000   # number of epochs to run
batch_size = 16 * 2 * 1  #10  # size of each batch
learning_rate=0.000001  #default 0.001
#torch.set_printoptions(8)
torch.set_printoptions(linewidth=140)
np.set_printoptions(linewidth=140)
#torch.set_default_dtype(torch.float64)

data_folder='data'
result_folder='checkpoints'
title='m4'
title='m6'
tag='v0'
gpu=0
single_data_file=False

exec(open('configurator.py').read()) # overrides from command line or config file
######################### config end   ###############################
LAYERS= [hidden_size for _ in range(num_hidden_layers+2)]
LAYERS[0]=16
LAYERS[-1]=16  # v (76, 108) from 76 to 92 for real part; imag is currently zero
note=f'{tag}-ReLU-Adam{learning_rate}-bs{batch_size}-layers{"_".join( str(_) for _ in LAYERS)}'
filename_prefix=f'{data_folder}/{title}'  #for loading data
filename_checkpoint=f'{result_folder}/{title}-{note}-check.pt'
filename_loss=f'{result_folder}/{title}-{note}-loss.pt'
#print('title/note:',title,note)
#print('input/output files:',filename_prefix,filename_checkpoint,filename_loss)

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str)) and k not in ['arg','key','val','attempt']]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
import json
print(json.dumps(config, indent=2))

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
# or use the following inside python
if device == "cuda":
    torch.cuda.set_device(gpu)
print(f"Using {device} device")


# check if one can get the energy from v and Ham
def verify_data(d):
    X = d[:,:16]      #Ham
    y = d[:,76:92]    #v
    Ene=d[:,32:36]    #Energy
    err = get_err(Ham, v, Ene)
    #err = get_err(X_val, y_pred, Ene_test)
    print(err)
    

import os
import glob
def load(filename_prefix): #loadd all files with this filename prefix
    filelist = glob.glob(f'{filename_prefix}*.npy')    
    print('get file list',filelist)
    print(f'loading {len(filelist)} data files...')
    data_list=[]
    for filename in filelist:
        _data=np.load(filename)
        assert _data.shape[1] == 108  # 42
        data_list.append(_data)
        if single_data_file==True:
            print('only processing',filename,'and skip other data files')
            break
    data = np.concatenate(data_list)

    if True:
        verify_data(data)
        exit()
    return data


# load training data
print(f'loading data: {filename_prefix}')
d = load(filename_prefix)
#d = d[:int(1e6)] # maximum 1 million data
d = torch.tensor(d,device=device)
print('sample entry d[0]')
print(d[0])

#torch.set_default_dtype(torch.float16)
d=d.float()  #differ by 1e-9
#d=d.half()  # half precision for fast training on large nn
X = d[:,:16]
#y = d[:,-10:]  #energy and parameters
#y = d[:,-6:]    #parameters
y = d[:,76:92]    #parameters

# use the last 1000 for evaluation
eval_size=d.shape[0]//5
X_test,y_test = X[-eval_size:],y[-eval_size:] 
X,y = X[:-eval_size],y[:-eval_size]

#Ene_test=d[-eval_size:,32:40] # to calculate accuracy
Ene_test=d[-eval_size:,32:36] # to calculate accuracy # real part only
#Ene_test = Ene_test.cpu()

print(type(X),X.dtype)
print('data shape X Y',X.shape,y.shape)
print('test shape X Y',X_test.shape,y_test.shape)


##  accuracy: reconstruct energy 
#loss_acc = nn.MSELoss()
def get_err(X_test,y_pred, Ene_test):
    #return torch.vdot(ve1, AA @ ve1)
    num = X_test.shape[0]
    v = X_test.reshape((num,4,4))
    Ham = y_pred.reshape((num,4,4))
    #print(torch.einsum('ij,j->i',AA , ve1)) # original Ham @ v, not in tensor/in parallel
    #Ene_pred =  torch.vdot(v, Ham @ v)
    Ham_v = torch.einsum('nij, nvj->nvi', Ham , v)  #Ham @ v
    Ene_pred =  torch.linalg.vecdot(v, Ham_v)

    e0=Ene_test.sum(dim=1)
    e1=Ene_pred.sum(dim=1)
    err = ( (e1-e0)/e0 ).abs().mean()
    #ratio = (e1/e0).mean()
    #acc = 1 - (1 - ratio).abs()
    err = err.detach().cpu().item()
    #acc = acc.detach().cpu().item()
    #acc = acc * 100 # percentage diff
    #print(e0[:10])
    #print(e1[:10])    
    #print(e0.shape,e1.shape,acc)
    #input()
    return err

#exit()
class Deep(nn.Module):
    def __init__(self,layers=[28*28,640,640,60,10]):
        super().__init__()
        def get_modules():
            modules=[]
            print('processing layers:',layers)
            num_layers=len(layers)
            for i in range(num_layers-2):
                layer0 = layers[i]
                layer1 = layers[i+1]
                layer = nn.Linear(layer0,layer1)
                act = nn.ReLU()
                #act = nn.LeakyReLU()
                modules.append(layer)
                modules.append(act)
                #modules.append(nn.Dropout(p=0.2))
            return modules
        self.linear_relu_stack = nn.Sequential(*get_modules())
        #self.linear_relu_stack = nn.Sequential(*modules)
        #parallel doesn't improve with identical components, skip it
        #self.parallel=Parallel(nn.Sequential(*get_modules()), nn.Sequential(*get_modules()),)
        #self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(layers[-2], layers[-1])
        
    def forward(self, x):
        x = self.linear_relu_stack(x)
        #x = self.parallel(x)
        #x = self.sigmoid(x)
        x = self.output(x)
        return x


def model_train(model, X_train, y_train, X_val, y_val,best_err=np.inf,best_weights = None):
    for i in [X_train, y_train, X_val, y_val]:
        print(i.shape)
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
                training_loss_list.append( loss.detach().cpu().item() )
                bar.set_postfix(loss=float(loss))
                #bar.set_postfix(loss=float(loss), best_acc = float(best_acc),acc=float(acc))
        #print('sample training loss:','\t'.join( [f'{_:.3f}' for _ in training_loss_list[-100:]]))
        training_loss = np.array(training_loss_list).mean()
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        loss = loss_fn(y_pred,y_val)
        _loss = loss.detach().cpu().item()
        err = get_err(X_val, y_pred, Ene_test)
        if err <  best_err:
            #best_acc = acc
            best_err = err
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights,filename_checkpoint)
            print(f'best weights saved into {filename_checkpoint} at epoch={epoch}, err={err}, loss={loss}')            
        #print(loss_list)
        loss_list.append( (_loss,training_loss, err, best_err, epoch) )
        loss_np_array = np.array(loss_list)
        print('  | validation  |  training |    err    | best_err   |   epoch :  history')
        print(loss_np_array[-10:])
        
        torch.save(loss_np_array,filename_loss)
        print(f'loss list saved into {filename_loss} {loss_np_array.shape} at loss={loss_np_array[-1]}')
        
            
        print('target:     ',end='')
        print((y_val)[0])
        print('prediction: ',end='')
        print(y_pred[0])
        print('diff:       ',end='')
        print((y_pred-y_val)[0])


    model.load_state_dict(best_weights)   # restore model and return best accuracy
    return best_acc,best_weights


model = Deep(LAYERS).to(device)
print(model)
print(f'batch_size={batch_size}')

# Hold the best model
best_err =  np.inf   # init to negative infinity
best_weights = None

# loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=0.001)


loss_list=[]
# load previous result for continuous training
if True:
    try:
        best_weights = torch.load(filename_checkpoint)
        model.load_state_dict(best_weights)

        loss_np_array = torch.load(filename_loss)
        loss_list = loss_np_array.to_list()
        #for i in range(loss_np_array.shape[0]):
        #    loss_list.append(  (loss_np_array[i,0],loss_np_array[i,1])   )
        model.eval()
        y_pred = model(X_test)
        loss = loss_fn(y_pred,y_test)
        best_err = get_err(X_test, y_pred, Ene_test)
        print('loaded previous weights with best_acc:',best_err,'loss_list length:',len(loss_list))
    except:
        print('did not find previous weights:',filename_checkpoint)

model_train(model, X, y, X_test, y_test, best_err, best_weights)

print(config)
print('program finished')

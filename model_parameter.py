frontmatter='''Train the nerual network
      Input: 16 random numbers f to generate the Hamiltonian
      Output: 16 ansatz parameters A
'''
print(frontmatter)

import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm
import torch
import os
import glob
from parallel import Parallel 

from data_generator_Parameter import get_err_F_array
#from configurator import print_config  #no need to import when the file is executed

######################### config start ###############################
hidden_size= 64 * 1 * 1
num_hidden_layers=8 # 5
n_epochs = 10000   # number of epochs to run
batch_size = 2 * 2 * 1  #10  # size of each batch
learning_rate=0.0001  #default 0.001
eps=1e-2   # small eplison in the denominator when calculating relative loss
#torch.set_printoptions(8)
torch.set_printoptions(linewidth=120)
np.set_printoptions(linewidth=120)

#data_folder='data'
data_folder='/public/home/weileizeng/ansatz-data/parameter/'
result_folder='checkpoints'
title='p1'
tag='v0'
gpu=0
#single_data_file=False
data_file_limit=-1
truncate_data_size=-1 #default -1
evaluation_only = False




exec(open('configurator.py').read()) # overrides from command line or config file
######################### config end   ###############################
LAYERS= [hidden_size for _ in range(num_hidden_layers+2)]
LAYERS[0]=16
LAYERS[-1]=16  # v (76, 108) from 76 to 92 for real part; imag is currently zero
#note=f'{tag}-ReL-Adam{learning_rate}-bs{batch_size}-layers{"_".join( str(_) for _ in LAYERS)}'
note=f'{tag}'
filename_prefix=f'{data_folder}/{title}'  #for loading data
filename_checkpoint=f'{result_folder}/{title}-{note}-check.pt'
filename_loss=f'{result_folder}/{title}-{note}-loss.pt'
filename_config_json=f'{result_folder}/{title}-{note}.json'
#print('title/note:',title,note)
#print('input/output files:',filename_prefix,filename_checkpoint,filename_loss)



#import wandb
#run = wandb.init(
#    # Set the project where this run will be logged
#    project="ACSE-universal-ansatz",
#    # Track hyperparameters and run metadata
#    config=config,
#)

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


from datetime import datetime
#now = datetime.now()
#python_date = now.strftime("%Y%m%d-%H%M%S")
python_date = datetime.now().__str__()


##  accuracy: reconstruct energy 
loss_err = nn.MSELoss()
@torch.no_grad()
def get_err(X_test,y_pred, Ene_test):
    #return torch.vdot(ve1, AA @ ve1)
    num = X_test.shape[0]
    Ham = X_test.reshape((num,4,4))
    v = y_pred.reshape((num,4,4))
    #print(torch.einsum('ij,j->i',AA , ve1)) # original Ham @ v, not in tensor/in parallel
    #Ene_pred =  torch.vdot(v, Ham @ v)   #for single entry n=1
    Ham_v = torch.einsum('nij, nvj->nvi', Ham , v)  #Ham @ v for n in parallel
    Ene_pred =  torch.linalg.vecdot(v, Ham_v)       # vdot for n in parallel

    # compare the sum
    # e0=Ene_test.sum(dim=1)
    # e1=Ene_pred.sum(dim=1)
    # compare all four term
    # e0=Ene_test 
    # e1=Ene_pred
    # compare the min (gorund state energy
    e0=Ene_test.min(dim=1)
    e1=Ene_pred.min(dim=1)
    #err = loss_err(Ene_pred, Ene_test)
    err = loss_err(e0,e1)

    Ene_pred_abs_mean = Ene_pred.sum(dim=1).abs().mean().item()

    print(f'Ene_pred_abs_mean={Ene_pred_abs_mean}')
    return err.detach().cpu()
'''
    print( ( (e1-e0)/e0 )  )
    print( ( (e1-e0)/e0 ).abs()  )
    print('_',_.abs().mean())
    print( ( (e1-e0)/e0 ).abs().mean()  )
    err = ( (e1-e0)/e0 ).abs().mean()
    err = err.detach().cpu().item()
    print('e0',e0[:10])
    print('e1',e1[:10])    
    print(e0.shape,e1.shape,err)
    return err
'''

# check if one can get the energy from v and Ham
def verify_data(d):
    d = torch.tensor(d,device=device)
    Ham = d[:,:16]      #Ham as X
    v = d[:,76:92]    #v as y
    Ene=d[:,32:36]    #Energy
    err = get_err(Ham, v, Ene)
    #err = get_err(X_val, y_pred, Ene_test)
    print(err)

mse = nn.MSELoss()
def get_loss(y0,y1,debug=False):
    loss1 = mse(y0,y1)
    loss2 = get_relative_loss(y0,y1,debug=debug)
    #print(f'loss1 = {loss1} (mse), loss2 = {loss2} (Relative)')
    gamma = 0.1
    loss = (loss1 + loss2 * gamma )/(1+gamma)
    if debug:
        print('losses:')
        print(loss1)
        print(loss2)
    return loss

# return relative loss between two data sets
# this enlarge effect on small values
# but fail when predition has a minus sign, which yields 1 always
def get_relative_loss(y0,y1,debug=False):
    r=(y1-y0).abs() / (y0.abs() + y1.abs() + eps)
    # check maximum of r
    if debug:
        #print('r',r)
        #print('r',r>0.5)
        #print('r',((r>0.5)*1.0).mean())
        index = torch.argmax(r)
        i,j = index//16, index % 16
        print('index,i,j',index,i,j)
        print(f'find max r[{index}] ={r[i][j]}, with {y0[i][j]}, {y1[i][j]} ')

        diff=mse(y0[i][j], y1[i][j] )
        print('mse(y0[i][j], y1[i][j] )=',diff)
        print('mse(y0[i], y1[i] )=',mse(y0[i], y1[i] ))
        print('mse(y0,y1)=',mse(y0,y1))
        print(y0.shape)
        print(y0[i])
        print(y1[i])
    #result = torch.max(r)
    #print(result)
#    print(v,index)

    return r.mean()
    

def load(filename_prefix): #loadd all files with this filename prefix
    filelist = glob.glob(f'{filename_prefix}*.npy')
    filelist.sort() #ensure the same validation data is used everytime
    if data_file_limit > 0 :
        filelist = filelist[:data_file_limit]
    #print('get file list (max 80)',filelist)
    print(f'loading {len(filelist)} data files...')
    data_list=[]
    for filename in filelist:
        print('loading...',filename)
        _data=np.load(filename)
        assert _data.shape[1] == 168 #108  # 42
        data_list.append(_data)
        #if single_data_file==True:
        #print('only processing',filename,'and skip other data files')
        #break
    data = np.concatenate(data_list)
    print('loaded data:',data.shape)

    if False:
        verify_data(data)
        exit()
    return data, filelist




# load training data
print(f'loading data: {filename_prefix}')
d, filelist = load(filename_prefix)
#d = d[:int(1e6)] # maximum 1 million data
#d = torch.tensor(d,device=device)
d = torch.tensor(d)
d=d.float()  #differ by 1e-9
d=d.to(device)
print('sample entry d[0]')
print(d[0])

#torch.set_default_dtype(torch.float16)
#d=d.half()  # half precision for fast training on large nn

if truncate_data_size>0: #control data size
    d=d[:truncate_data_size]



X = d[:,:16]  #random input for Ham
y = d[:,16:32]    # Ansatz parameter



# use the last 1000 for evaluation
eval_size=d.shape[0]//5


X_test,y_test = X[-eval_size:],y[-eval_size:] 
X,y = X[:-eval_size],y[:-eval_size]
Xyshape=[X.shape,y.shape]  # record data size

#Ene_test=d[-eval_size:,32:40] # to calculate accuracy
Ene_test=d[-eval_size:,88:92] # to calculate accuracy # real part only
#Ene_test = Ene_test.cpu()
Ene_abs_mean = Ene_test.sum(dim=1).abs().mean().item()

print(type(X),X.dtype)
print('data shape X Y',X.shape,y.shape)
print('test shape X Y',X_test.shape,y_test.shape)


A_flat_test = d[-eval_size:,16:32] # used in evaluation only mode


if True:
    # to limit eval size for saving time
    max_eval_length = 10000
    X_test = X_test[:max_eval_length]
    y_test = y_test[:max_eval_length]
    Ene_test = Ene_test[:max_eval_length]
    A_flat_test = A_flat_test[:max_eval_length]
del d # to save some memory


import signal
# print config before exit
def handler(signum, frame):
    print('*'*30,'ctrl-c received ...')
    top_frame = frame
    while top_frame.f_back:
        #print('getting back')
        top_frame = top_frame.f_back    
    print_config(top_frame.f_globals,additional_keys=additional_keys)
    #print('finish printing config')
    #signal.default_int_handler(signum,frame) # exit and print trace
    exit(1)
signal.signal(signal.SIGINT, handler)


additional_keys=['filelist','LAYERS']
config_keys, config = print_config(globals(),additional_keys = additional_keys)

#config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str)) and k not in ['arg','key','val','attempt']]
#config_keys.extend(['filelist','LAYERS'])
#config = {k: globals()[k] for k in config_keys} # will be useful for logging
import json
#print(json.dumps(config, indent=2))

# verify existing config file or create new one.
if os.path.exists(filename_config_json):
    print(f'Found existing config file: {filename_config_json}')
    if True:
      with open(filename_config_json, 'r') as f:
        # if file already exist, verify the config are the same
        cfg = json.load(f)
        #important_keys =['LAYERS','hidden_size']
        important_keys =['hidden_size']
        for k in important_keys:
            assert config[k] == cfg[k]
        #for k in config_keys:
        for k,_ in cfg.items():
            if config[k] != cfg[k]:
                print(f'Changed config: [{k}]\t  {cfg[k]} \t-> {config[k]}')        

if not evaluation_only:
    with open(filename_config_json, 'w') as f:
        json.dump(config, f,indent=2)
    print(f'config saved/overrided into {filename_config_json}')




#X_test = X_test.to(device)
#y_test = y_test.to(device)
#Ene_test=Ene_test.to(device)
# rewrite data using DataLoader

from torch.utils.data import Dataset,DataLoader
class MyDataset(Dataset):
    def __init__(self, X,y):
        self.X=X
        self.y=y
        self.length=X.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        _X=self.X[idx]
        _y=self.y[idx]
        return _X,_y

torch.multiprocessing.set_start_method('spawn')

#training_data=MyDataset(X,y)
#test_data=MyDataset(X_test,y_test)
#train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True,num_workers=0)
#test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


class ResBlock(nn.Module):
    def __init__(self, block):
        super(ResBlock, self).__init__()
        self.block = block
    def forward(self, x):
        return x + self.block(x)

#exit()
class Deep(nn.Module):
    def __init__(self,layers=[28*28,640,640,60,10]):
        super().__init__()
        def get_modules():
            modules=[]
            print('processing layers:',layers)
            num_layers=len(layers)
            for i in range(num_layers-2):
                layer0,layer1 = layers[i:i+2]                            
                modules.append(nn.Linear(layer0,layer1))
                #modules.append(nn.BatchNorm1d(layer1,eps=1e-08))
                #modules.append(nn.ELU())
                modules.append(nn.Tanh())
                #modules.append(nn.Dropout(p=0.5))
                
                #layer = nn.Linear(layer0,layer1)
                #bn=nn.BatchNorm1d(layer1,eps=1e-08)
                #act = nn.ReLU()
                #act = nn.ELU()
                #act = nn.LeakyReLU()
                #modules.append(layer)
                #modules.append(bn)
                #modules.append(act)
                #modules.append(nn.Dropout(p=0.2))
            return modules
        #self.linear_relu_stack = nn.Sequential(*get_modules())

        #rnn = nn.RNN(64,64,5)        
        #self.l1=nn.Linear(layers[0],64)
        #self.l2=rnn
        #self.l3=nn.Linear(64,layers[-1])


        # def conv2d block
        def get_conv2d(out_size):
            c=16
            return Parallel(
                nn.Sequential(nn.Flatten(),nn.Linear(4*4,out_size)),                
                nn.Sequential(nn.Conv2d(1,c,2),nn.Flatten(),nn.Linear(c*3*3,out_size)),
                nn.Sequential(nn.Conv2d(1,c,3),nn.Flatten(),nn.Linear(c*2*2,out_size)),
                merge='sum'
            )             
            
        def get_blocks():
            blocks=[]
            blocks.append(nn.Linear(layers[0],layers[1]))
            num_layers=len(layers)
            # [2,8,8,8,2]
            for i in range(1,num_layers-2):
                _block=[]
                layer0,layer1 = layers[i:i+2] # the i th and i+1 th elements
                _block.append(nn.Tanh())
                _block.append(nn.Linear(layer0,layer1))
                block=ResBlock(nn.Sequential(*_block))
                blocks.append(block)
            blocks.append(nn.Tanh())
            blocks.append(nn.Linear(layers[-2],layers[-1]))
            #one can add extra linear layer to propogate the residual
            return blocks
        

        #self.conv2d = get_conv2d(layers[1])
        
        self.blocks= nn.Sequential(*get_blocks())
        
        
    def forward(self, x):
        #x = self.blocks(x)

        #x = x.reshape(-1,1,4,4)
        #x = self.conv2d(x)
        x = self.blocks(x)
        
        return x


def model_train(model, X_train, y_train, X_val, y_val,best_err=np.inf,best_weights = None):
    for i in [X_train, y_train, X_val, y_val]:
        print(i.shape)
    val_split=False #split X_val during inference to save memory
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
                #X_batch,y_batch = next(iter(train_dataloader)) #dataloader is slower in our case when all data can be saved in GPU
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                #X_batch,y_batch = X_batch.to(device),y_batch.to(device)
                #print('X_batch.shape',X_batch.shape)
                
                # forward pass
                y_pred = model(X_batch)
                #print(y_pred.shape,y_batch.shape)
                #loss = loss_fn(y_pred, y_batch)
                #loss = get_reletive_loss(y_batch,y_pred)
                loss = get_loss(y_batch,y_pred)
                    
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
        # since the test size is large. use no_grad to save a lot of GPU memory

        with torch.no_grad():

            if not val_split:
                try:
                    y_pred = model(X_val)
                except:
                    print('run out of memory during validation.', 'X_val.shape', X_val.shape, 'run validation in split instead')            
                    val_split = True
                    val_batch_start = torch.arange(0, len(X_val), batch_size)
            if val_split:                
                ys=[]
                with tqdm.tqdm(val_batch_start, unit="batch", mininterval=0, disable=False) as bar:
                    bar.set_description(f"validating")
                    for start in bar:
                        X_batch = X_val[start:start+batch_size]
                        #y_batch = y_val[start:start+batch_size]
                        y_pred = model(X_batch)
                        ys.append(y_pred)
                        #bar.set_postfix(loss=float(loss))
                y_pred = torch.cat(ys)                                    
            #y_pred = model(X_val)
            #loss = loss_fn(y_pred,y_val)
            #loss = get_reletive_loss(y_val,y_pred)
            loss = get_loss(y_val,y_pred,debug=True)
            _loss = loss.detach().cpu().item()
            #err = get_err(X_val, y_pred, Ene_test)
            err = get_err_F_array(X_val, y_pred, Ene_test, device=device)
            #err = _loss
            if err <  best_err:
                #best_acc = acc
                best_err = err
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(best_weights,filename_checkpoint)
                print(f'best weights saved into {filename_checkpoint} at epoch={epoch}, err={err}, loss={loss}')            
        #print(loss_list)
        #wandb.log(dict(validation_loss=_loss,training_loss=training_loss, error=err, best_error=best_err, epoch=epoch))
        loss_list.append( (_loss,training_loss, err, best_err, epoch) )
        loss_np_array = np.array(loss_list)
        print('  |  validation  |  training  |    err     |   best_err   |   epoch   |  - history -')
        print(loss_np_array[-10:])
        
        torch.save(loss_np_array,filename_loss)
        print(f'loss list saved into {filename_loss} {loss_np_array.shape} at loss={loss_np_array[-1]}, Ene_abs_mean={Ene_abs_mean}')

        #Ene_val_abs_mean = Ene_test.sum(dim=1).abs().mean().item()
            
        print('target:     ',end='')
        print((y_val)[0])
        print('prediction: ',end='')
        print(y_pred[0])
        print('diff:       ',end='')
        print((y_pred-y_val)[0])


    model.load_state_dict(best_weights)   # restore model and return best accuracy
    return best_err,best_weights


model = Deep(LAYERS).to(device)
print(model)
print(f'batch_size={batch_size}')

# Hold the best model
best_err =  np.inf   # init to negative infinity
best_weights = None

# loss function and optimizer
#loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)


loss_list=[]
# load previous result for continuous training
if os.path.exists(filename_checkpoint):
        best_weights = torch.load(filename_checkpoint)
        model.load_state_dict(best_weights)

        loss_np_array = torch.load(filename_loss)
        loss_list = loss_np_array.tolist()
        #print(loss_list)
        #for i in range(loss_np_array.shape[0]):
        #    loss_list.append(  (loss_np_array[i,0],loss_np_array[i,1])   )
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test[:batch_size])
            #loss = loss_fn(y_pred,y_test)
            #best_err = get_err_F_array(X_test[:batch_size], y_pred, Ene_test[:batch_size],device=device)
            _loss = get_loss(y_test[:batch_size],y_pred, debug=True).detach().cpu().item()            

            #_loss = get_reletive_loss(y_test[:batch_size],y_pred).detach().cpu().item()            
            err = _loss

            #best_err = get_err(X_test[:batch_size], y_pred, Ene_test[:batch_size])
            print('loaded previous weights with best_acc:',best_err,'loss_list length:',len(loss_list))
            # an estimate on best acc, in fact one can read from data file. best_err = loss_np_array[-1,4]

            
            print('X_test[:batch_size]:',X_test[:batch_size].shape)
            print(y_pred.shape)
            best_err = get_err_F_array(X_test[:batch_size], y_pred, Ene_test[:batch_size],device=device)
            print('err on predition:',best_err.item())
            # to verify the program is right
            data_err = get_err_F_array(X_test[:batch_size], A_flat_test[:batch_size], Ene_test[:batch_size],device=device)
            print('err on saved data err is:',data_err.item())     
            print('(y_pred - A_flat_test).mean()=', (y_pred - A_flat_test[:batch_size]).mean().item() )         

            if evaluation_only:
                exit()
else:
    print('did not find previous weights:',filename_checkpoint)

model_train(model, X, y, X_test, y_test, best_err, best_weights)

print(config)
print('program finished')

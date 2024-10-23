print('''
      
machine learning ansatz for hubbard model
      
''')

import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm
import torch

#config
#L=5
#folder='../../DMRG/tenpy/data'
folder='/data/zwl/hubbard'
#filename=f'{folder}/data-ising-L{L}-1.pt'  # 41450 entries
#filename=f'{folder}/data-ising-L{L}-2.pt'  # 84950 entries
#filename=f'{folder}/data-ising-L5.dict.pt.array'

filename=f'{folder}/eigen.npy'
filename=f'{folder}/eigen100000.npy'
filename=f'{folder}/h10-0.npy'
_=filename.split('/')[-1]
filename_checkpoint=f'result/{_}.32'
filename_loss=f'result/{_}.loss.32'
print('input/output files:',filename,filename_checkpoint,filename_loss)

# config
#trials=30
#output_width=95-9
output_width=11
hidden_size= 16
num_hidden_layers=2
LAYERS= [hidden_size for _ in range(num_hidden_layers+2)]
LAYERS[0]=10
LAYERS[-1]=output_width
#LAYERS=[2*L-1,L*8*8,L*8*8,L*8*8,L*8*8,1]
n_epochs = 50 #250   # number of epochs to run
batch_size = 64*8 #10  # size of each batch
#torch.set_printoptions(8)
torch.set_printoptions(linewidth=140)

#torch.set_default_dtype(torch.float64)

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

print('loading data...')
#d = np.load('eigen.npy')
d = np.load(filename)
d = torch.tensor(d)
print(d)


#d = torch.load(filename)

#truncation = 1e-15

#d = d * (d.abs()>truncation) 

print('sample entry d[0]')
print(d[0])
#d1=d[1]
d=d.float()  #differ by 1e-9
#d2=d[1]
#print(d[1])

#print(d1-d2.double())
X = d[:,:10]
y = d[:,12:22]
#y=y.reshape((len(y),1))
#X = d['X']
#y = d['y']
# X = X[:10000] #achieve same acc using 10000 entries instead of 40000 entries
# y = y[:10000]
print('y',y)

print('data shape X Y',X.shape,y.shape)
print(type(X),X.dtype)
#torch.save(data,filename)
#X_test,y_test = X[:1000],y[:1000]
#print('test shape X Y',X_test.shape,y_test.shape)

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

# check the percentage error in predicted output ( ground state energy)
def acc_eval(y_pred,y_batch):
    #print(torch.sqrt(((y_pred - y_batch)**2).mean()))
    #print(torch.sqrt(((y_pred - y_batch)**2).mean())/ (y_batch))
    #print(torch.sqrt(((y_pred - y_batch)**2).mean())/ (y_batch).mean() )
    #print(torch.sqrt(((y_pred - y_batch)**2).mean())/ ((y_batch).mean()) )
    #print(y_pred.mean(0))
    #print(y_batch.mean(0))
    #print( torch.sqrt(((y_pred - y_batch)**2) ) )
    #print( torch.sqrt(((y_pred - y_batch)**2) ).mean(0)  )
    #print(y_batch.mean(0))
    delta_mean=torch.sqrt(((y_pred - y_batch)**2) ).mean(0)
    y_batch_mean = y_batch.mean(0)
    y_pred_mean  = y_pred.mean(0)
    y_mean = y_batch_mean + y_pred_mean
    #print(delta_mean/y_mean)
    #print(  (delta_mean/y_mean).mean() )
    #input()
    acc = (delta_mean/y_mean).mean()
    if acc > 0 :
        acc = - acc
    return  acc
#return  torch.sqrt(((y_pred - y_batch)**2).mean())/ ((y_batch).mean())




def model_train(model, X_train, y_train, X_val, y_val,best_acc=-np.inf,best_weights = None):
    for i in [X_train, y_train, X_val, y_val]:
        print(i.shape)
    # loss function and optimizer
    ##loss_fn = nn.BCELoss()  # binary cross entropy
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()
    loss_list=[]
    optimizer = optim.Adam(model.parameters(), lr=0.0001)


    batch_start = torch.arange(0, len(X_train), batch_size)


    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
            bar.set_description(f"Epoch {epoch}/{n_epochs}")            
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                X_batch,y_batch = X_batch.to(device),y_batch.to(device)
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
        X_val=X_val.to(device)
        y_val=y_val.to(device)        
        y_pred = model(X_val)
                #acc = ((y_pred>0) == y_val).type(torch.float).mean()
        #acc = acc_eval(y_pred,y_val)
        loss = loss_fn(y_pred,y_val)
        loss_list.append(loss)
        #torch.save(loss_list,filename_loss)
        #print(f'loss list saved into {filename_loss}')
        acc = - loss
        #print( ((y_pred-y_val)/y_val).abs() )
        
        #print(y_pred)
        #print(y_pred-y_val)
        #print(y_val)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
            #save into file
            #torch.save(best_weights,filename_checkpoint)
            #print(f'weights saved into {filename_checkpoint} at epoch={epoch}, acc={acc}')
    #skip best acc
    #return acc
    # restore model and return best accuracy
    model.load_state_dict(best_weights)

    import matplotlib.pyplot as plt
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.title('training for energy vs u \n Hamiltonian input: (1,1,1,1,1,u/2,u/2,u/2,u/2,u/2)')
    #plt.plot(range(0,10000+1)/10,data)
    #x = np.linspace(start:=0,stop:=1.0001/2,num=10001)
    #print(loss_list)
    _ = torch.tensor(loss_list).cpu()
    print(_)
    #plt.plot(loss_list.cpu())
    plt.yscale('log')
    plt.plot(_,label='loss')
    plt.show()

    return best_acc,best_weights

#from sklearn.model_selection import StratifiedKFold, train_test_split

cv_scores = []
#model = Deep().to(device)
#for train, test in kfold.split(X,y[:,1]):

# train the same model the the same data a few times
layers=LAYERS    
model = Deep(layers).to(device)
print(model)


# Hold the best model
best_acc = - np.inf   # init to negative infinity
best_weights = None

for i in range(1):
    perm = torch.rand
    indices = torch.randperm(X.size()[0])
    X=X[indices]
    y=y[indices]
    X_test,y_test = X[-1000:],y[-1000:]
    _X,_y = X[:-1000],y[:-1000]
    #modify test data set as well    
    #acc = model_train(model, X[train], y[train], X[test], y[test])
    best_acc,best_weights = model_train(model, _X, _y, X_test, y_test, best_acc, best_weights)
    # restore model and return best accuracy

    model.load_state_dict(best_weights) 
    
    acc=best_acc
    print("Accuracy (wide): %.8f" % acc)
    cv_scores.append(acc.detach().cpu())
    #break
    
# evaluate the model
print('historical acc',cv_scores)
cv_scores=np.array(cv_scores)
#acc = np.mean(cv_scores)
#std = np.std(cv_scores)
#print("Model accuracy: %.2f%% (+/- %.2f%%)" % (acc*100, std*100))

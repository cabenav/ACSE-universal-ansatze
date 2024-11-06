print('''     
machine learning ansatz for hubbard model     
''')

import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm
import torch
import signal
from configurator import handler
signal.signal(signal.SIGINT, handler) # print config before exit

# CONFIG
gpu=6  # the indexed gpu to be used
folder='/data/zwl/hubbard'  #data folder


L=8 # to decide data shape and position
if L==5:
    #filename=f'{folder}/h10-0.npy'  # L=5
    filename=f'{folder}/L5n2-h10-0.npy'  # L=5， added decoherence data
    #tag='-20000-v0.3'
    #tag='-2000-v1.4'  
    #tag='-200-v0.4'
    #tag='-20-v0.8'  #tag='-20-v0.5'

elif L==8:
    #filename=f'{folder}/L8n2-h10-0.npy'  # L=8
    filename=f'{folder}/L8n2-h10-wd-0.npy'  # L=8, added decoherence data
    tag='-2000-v1.2'
    tag='-20000-v1.2'

truncate_data_size=20000 #default -1
eval_size_min = 10
eval_size_max = 100

_=filename.split('/')[-1]  # folder/name.npy -> name
_=_+tag
filename_checkpoint=f'results/{_}.32'
filename_loss=f'results/{_}.loss.32'
print('input/output files:',filename,filename_checkpoint,filename_loss)
filename_config_json=f'results/{_}.json'

# Nerual network hyper paramters
if L==5:
    input_width=10
    output_width=10
elif L==8:
    input_width = L*2
    output_width = 28
hidden_size= 64
num_hidden_layers=3
LAYERS= [hidden_size for _ in range(num_hidden_layers+2)]  # 64 x 3
LAYERS[0]=input_width
LAYERS[-1]=output_width

learning_rate  = 0.00001  #0.0001
n_epochs = 11200 #250   # number of epochs to run
batch_size = 8*1 #10  # size of each batch


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
# or use the following inside python
if device == "cuda":
    torch.cuda.set_device(gpu)
print(f"Using {device} device with gpu={gpu}")

print('loading data...') 
#d = np.load('eigen.npy')
data = np.load(filename)
data = torch.tensor(data)

# Adjust evaluation dataset size
eval_size=data.shape[0]//5 #use 20% for test
if eval_size < eval_size_min:
    evale_size = eval_size_min
if eval_size > eval_size_max:
    eval_size = eval_size_max

# Adjust dataset size
#d = d[:20000] # limit data
data = data[:(truncate_data_size+eval_size)] # limit data
#print(data)

#eval_size_end = eval_size + 1000 # default -1

print('sample data entry data[0]')
print(data[0])
data=data.float()  #differ by 1e-9   # float is much faster and usually accurate enough; to use double, comment this line and set default type to be 

# Get training data pair and evaluation data
if L==5:
    X = data[:,:10]
    #y = d[:,11:22] #with energy
    y = data[:,12:22]
elif L==8:
    X = data[:,:L*2]
    _index = 2*L +1+1
    y = data[:,_index:(_index+28)]
#energy_from_file = data[:,11]
energy_from_file = data[:,-1]
obsevables_from_file = data[:,-2:]

print('data shape X Y ob',X.shape,y.shape,obsevables_from_file.shape)
print(type(X),X.dtype)
#input()

from configurator import print_config, save_config
#additional_keys=['filelist','LAYERS']
#config_keys, config = print_config(globals(),additional_keys = additional_keys)
config_keys, config = print_config(globals())
save_config(config, filename_config_json)


from data_generator_Hubbard import Xy2energy, Xy2acc
mse = nn.MSELoss()
np.set_printoptions(linewidth=140)
if L==5:
    compute_acc = Xy2acc

def get_acc(energy_data,X,y):
    n = energy_data.shape[0]
    energy_computed = torch.zeros_like(energy_data)
    X = X.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    for i in range(n):
        _X = X[i]
        _y = y[i]
        _energy = Xy2energy(_X,_y)
        energy_computed[i] = _energy
    acc = - mse(energy_data,energy_computed)
    if False:  #print for test
        print(energy_data)
        print(energy_computed)
        print(energy_computed- energy_data)
        print('acc=',acc)
        input('...')
    return acc

def get_acc2(observables_data,X,y):
    '''
        all input are n-row tensors
        observables should have two columns for decoherence and energy respectively
    '''
    #print('observables_data.shape',observables_data.shape)
    #input('...')
    n = observables_data.shape[0] # get the batchsize
    observables_computed = torch.zeros_like(observables_data)
    X = X.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    for i in range(n):
        _X = X[i]
        _y = y[i]
        #print(observables_data)
        _energy, _decoherences_approx = Xy2acc(_X,_y,decoherences_data= observables_data[i,0].item())
        observables_computed[i,0] = _decoherences_approx
        observables_computed[i,1] = _energy
    acc_decoherence = - mse(observables_data[:,0],observables_computed[:,0])
    acc_energy = - mse(observables_data[:,1],observables_computed[:,1])
    if False:  #print for test
        print(observables_data)
        print(observables_computed)
        print(observables_computed- observables_data)
        print('acc=',acc_decoherence,acc_energy)
        input('...')
    return acc_decoherence,acc_energy

# verify data here
#get_acc(energy,X,y+1e-4)
#input('...')

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

def model_train(model, X_train, y_train, X_val, y_val,best_acc_energy=-np.inf,best_weights = None):
    for i in [X_train, y_train, X_val, y_val]:
        print(i.shape)
    # loss function and optimizer
    loss_fn = nn.MSELoss()
    loss_list=[]    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    batch_start = torch.arange(0, len(X_train), batch_size)

    loss_ansatz=1.0 # init value
    acc_energy = -10
    best_acc_energy = acc_energy
    acc_decoherence = -10
    best_acc_decoherence = acc_decoherence
    acc_ref = acc_ref0

    X_val=X_val.to(device)
    y_val=y_val.to(device)  # use the same evaluation set throughout all epoches

    for epoch in range(n_epochs):
        model.train()
        loss_train_list=[]
        if True: # whether to shuffle the data
            # permutate input data order randomly
            indices = torch.randperm(X_train.size()[0])
            X_train=X_train[indices]
            y_train=y_train[indices]
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
                loss_train = loss_fn(y_pred, y_batch)
                #loss_train = loss_fn(y_pred, y_batch) + loss_fn(y_pred[:,0],y_batch[:,0]) * 1000.0  # 10 times attention on energy
                loss_train_list.append(loss_train)
                # backward pass
                optimizer.zero_grad()
                loss_train.backward()
                # update weights
                optimizer.step()
                # print progress                
                bar.set_postfix(
                    loss_ansatz=float(loss_ansatz),
                    best_acc_energy = float(best_acc_energy),
                    acc_energy=float(acc_energy),
                    loss_train=float(loss_train),
                    acc_ref = float(acc_ref)  # added for test
                )
        loss_train_mean = torch.tensor(loss_train_list).cpu().mean()
        # evaluate accuracy at end of each epoch
        model.eval()
                
        y_pred = model(X_val)
        loss_ansatz = loss_fn(y_pred[:,1:],y_val[:,1:]) # on ansatz

        # need compute energy here
        print('compute obsevables use acc2')
        #acc_decoherence
        #acc_energy,_ =  get_acc2(obsevables_val,X_val,y_pred)
        #print(obsevables_val)
        acc_decoherence,acc_energy = get_acc2(obsevables_val,X_val,y_pred)
        print('-'*50,'acc_energy=',acc_energy,'acc_decoherence=',acc_decoherence)
        #acc_ref = get_acc(energy_val,X_val,y_val)

        #print(y_pred)
        #print(y_pred-y_val)
        #print(y_val)
        if acc_decoherence > best_acc_decoherence:
            best_acc_decoherence = acc_decoherence
        if acc_energy > best_acc_energy:
            best_acc_energy = acc_energy
            best_weights = copy.deepcopy(model.state_dict())
            #save into file
            #torch.save(best_weights,filename_checkpoint)
            #print(f'weights saved into {filename_checkpoint} at epoch={epoch}, acc={acc}')
        loss_list.append([loss_train_mean,loss_ansatz,-acc_energy,-best_acc_energy,-acc_decoherence,-best_acc_decoherence])

        if (1+epoch) % 10 ==0: # save loss for each 100 epoches
            _loss2 = torch.tensor(loss_list).cpu()
            np.save(filename_loss,_loss2)
            print(f'epoch = {epoch}, loss saved into {filename_loss}')

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    
    loss2 = torch.tensor(loss_list).cpu()
    #np.save('results/hubbard_loss.npy',loss2)
    np.save(filename_loss,loss2)

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

    return best_acc_energy,best_weights


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
    obsevables_from_file=obsevables_from_file[indices]
    X_test,y_test = X[-eval_size:],y[-eval_size:]
    _X,_y = X[:-eval_size],y[:-eval_size]
    energy_from_file = energy_from_file[indices]
    energy_val = energy_from_file[-eval_size:]
    obsevables_val = obsevables_from_file[-eval_size:]
    print('obsevables_val:',obsevables_val)


    #acc_ref0 = get_acc(energy_val,X_test,y_test) # test past, good data
    _,acc_ref0 = get_acc2(obsevables_val,X_test,y_test) # test past, good data



    #X_test,y_test = X[-1000:],y[-1000:]
    #_X,_y = X[:-1000],y[:-1000]
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

print_config(globals())

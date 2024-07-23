import torch
import numpy as np
import tqdm

# the function to generate one data entry
from data_generator_Parameter import generate, get_err_F_array

print('This program run generate() in parallel, and save into data folder incrementally')

############################# config start #################################
trials = 10000
# scipy use defult omp threads=4 for each process. hence the actual cpu being used is num_threads * 4. total core 86
num_threads=12  # 86 - 12*4 = 38 # 86-24=62 /4 = 15
# block to save data
block_size = num_threads * 200 * 5
#folder = 'data'
folder = '/public/home/weileizeng/ansatz-data/parameter'
#filename_prefix=f'{folder}/m4'
#filename_prefix=f'{folder}/m6'
filename_prefix=f'{folder}/p1'
#filename='tmp.npy'
# discontribute data into list of files with limited filesize or avoid slow I/O
filesize_limit = 300 #50 # in Mb
filesize_limit_in_bytes = filesize_limit * 1.0e6 # in bytes

TEST=True
############################# config end  #################################
from configurator import print_config
print_config(globals())

import os
def append(filename_prefix, array , filesize_limit = 100.0):
    '''append array to given file sets'''
    data = array
    for i in range(1000):        
        filename = f'{filename_prefix}-{i}.npy'
        if os.path.exists(filename):
            if (os.path.getsize(filename) > filesize_limit_in_bytes):
                continue
            else:
                data_old = np.load(filename)
                data = np.concatenate((data_old, array))            
        np.save(filename, data)
        return filename, data.shape
    #print(f'{trial}/{trials} data {data.shape} appended into {filename} {data_new.shape}')

    
import sys
from multiprocessing import Pool
if __name__=="__main__":
    if TEST:
        
        import torch
        device='cuda'
        if False:
            row = generate(1)        
            #print('_'*80)
            #generate(1)
            #exit()
            
            _data = torch.tensor(row)
            data=torch.tensor(np.array([_data]))
            data=data.to(device)
        #verify_data(data)

        def error_test(data):
            # input: random numbers for Ham
            # output: exsatz parameter
            # metric: energy

            d=data
            
            X = d[:,:16]      # f
            y = d[:,16:80]     # A
            #print(y)
            #exit()
            Ene_test=d[:,88:92]
            #def get_err(X_test,y_pred, Ene_test):
            print('_'*80)
            #get_err_F(X,y, Ene_test)
            get_err_F_array(X,y, Ene_test,device='cuda')
        #exit()
        block_size=1
        with Pool(num_threads) as p:
            result = list(tqdm.tqdm(p.imap(generate, range(block_size)), total=block_size))
            #result = p.map(generate,list(range(block_size)))
            data = np.array(result)
        # program end without saving into file

        #print('data:',data)
        error_test(torch.tensor(data,device=device))
        exit()
        
        verify_data(data)

        exit()

    for trial in range(trials):
        with Pool(num_threads) as p:
            result = list(tqdm.tqdm(p.imap(generate, range(block_size)), total=block_size))            
            #result = p.map(generate,list(range(block_size)))
            data = np.array(result)
            filename, shape = append(filename_prefix,data)
            print(f'[{trial}/{trials}] data {data.shape} appended into {filename} {shape}')
                

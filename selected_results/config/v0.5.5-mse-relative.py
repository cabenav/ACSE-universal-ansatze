description = "change to two loss term: mse + relative; use gamma = 0.1; small batch size"
#tag="v0.4.0-mse-relative-v3" 
# data in region (-0.2, 0.2)
data_folder='/public/home/weileizeng/ansatz-data/p2'
title='p2'
tag=config_file.split('/')[-1][:-3] # config/v4.py -> v4
batch_size=64 
num_hidden_layers=6 
hidden_size=256  
data_file_limit=20 
learning_rate=0.0001 
eps=1e-2
#evaluation_only=True
gpu=2

# add MSE loss
comment = """
    Small batch size helps, 64 behave much better than 256. should try 16
    change to data in new region
    remove conv2d
    modify relative loss
    remove relative loss
    add more data
"""
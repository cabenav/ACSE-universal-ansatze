description = "change to two loss term: mse + relative; use gamma = 0.1; small batch size"
# data_folder='/public/home/weileizeng/ansatz-data/p2'  # old foler in qlab 8
data_folder='/data/zwl/ansatz-data/p2'  # folder in A100x8
title='p2'  #f5, f9, f38, p2
tag=config_file.split('/')[-1][:-3] # config/name.py -> name
batch_size=2 
num_hidden_layers=6 
hidden_size=256  
data_file_limit=1 #1, -1, 20,
truncate_data_size=20 #default -1
learning_rate=0.0001 
eps=1e-2  # small eplison in the denominator when calculating relative loss
#evaluation_only=True
gpu=6

# add MSE loss
comment = """
    Small batch size helps, 64 behave much better than 256. should try 16
    change to data in new region
    remove conv2d
    modify relative loss
    remove relative loss
    add more data
    data ranged (-0.2,-0.2) 
    more data
    now try a test with minimal data
"""
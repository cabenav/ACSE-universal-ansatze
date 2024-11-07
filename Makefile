GPU=0 #default gpu to use. can also be changed in .py file

.PHONY: data train plot plot-all test config hubbard
#python3.10 -m venv env
#source env/bin/activate
py	=env/bin/python

all:config

eval:my-model_parameter_eval

CONF=config/v0.4.0-mse-relative-v2.py
CONF=config/v0.4.0-mse-relative.py
CONF=config/v0.4.0-mse-relative-v3.py
CONF=config/v0.4.0-mse-relative-v4.py
CONF=config/v0.4.0-mse-relative-v5.py
CONF=config/v0.4.1-mse-relative.py
CONF=config/v0.4.2-test-mse-relative.py
CONF=config/v0.4.3-conv2d-mse-relative.py
CONF=config/v0.5.7-mse-relative.py
CONF=config/v0.8.1-mse-relative.py
CONF=config/v0.8.2-mse-relative.py
CONF=config/v0.8.3-mse-relative.py  #batchsize 64
CONF=config/v0.8.4-mse-relative.py  #batchsize 8
CONF=config/v0.8.5-mse-relative.py  #batchsize 4
CONF=config/v0.8.6-mse-relative.py  #200k data
CONF=config/v0.8.7-mse-relative.py  #20 data
#config/v0.3.2-base.py

# train the spin model with given config
config:
	$(py) model_parameter.py ${CONF}

# train with new data 16->16 mapping
my-model_parameter_eval:
	$(py) model_parameter.py \
--tag="v0.3.0-min-F-rnn" \
--batch_size=256 \
--num_hidden_layers=6 \
--hidden_size=256  \
--data_file_limit=2 \
--learning_rate=0.0001 \
--gpu=5 \
--evaluation_only=True


# train with new data 16->16 mapping
my-model_parameter:
	$(py) model_parameter.py \
--tag="v0.3.5-min-F-rnn" \
--batch_size=1256 \
--num_hidden_layers=6 \
--hidden_size=128  \
--data_file_limit=40 \
--learning_rate=0.0001 \
--eps=1e-2 \
--gpu=2


#--tag="v0.3.1-min-F-rnn" \
#eps=1e-5

# change to relative loss
#	$(py) model_parameter.py \
--tag="v0.3.0-min-F-rnn" \
--batch_size=256 \
--num_hidden_layers=6 \
--hidden_size=256  \
--data_file_limit=40 \
--learning_rate=0.0001 \
--gpu=4

# train for ansatz parameter 
my-model_F:
	$(py) model_F.py \
--tag="v0.2.4-min-F-rnn" \
--batch_size=4256 \
--num_hidden_layers=6 \
--hidden_size=256  \
--data_file_limit=20 \
--learning_rate=0.0001 \
--gpu=4


# train for the exponent v
my-model:
	$(py) model.py \
--tag="v0.2.0-rnn" \
--batch_size=4256 \
--num_hidden_layers=6 \
--hidden_size=1024  \
--data_file_limit=-1 \
--learning_rate=0.0001 \
--gpu=7
#--single_data_file=False \
#	$(py) model.py --tag="v6-dropout0.2-sigmoid" --batch_size=1024 --num_hidden_layers=6 --gpu=4
#	$(py) model.py --tag="v5-parallel" --batch_size=128 --num_hidden_layers=6 --gpu=3

# tune tip
# no improvement from dropout, parallel(identical), sigmoid?
# large batch_size=1024 seems to be fine

data:
	$(py) data_generator_Pool.py
#	$(py) data-generator-Mqubit.py

model:
	CUDA_VISIBLE_DEVICES=${GPU} $(py) model.py ${argv}

	
#make f=<> plot
plot:
	$(py) plot.py ${f}
plot-all:
	$(py) all-plots.py
nvtop:
	nvidia-smi |head -n 15

#sync figs to weilei's macbook
rsync:
	rsync -rP root@10.200.69.64:/root/weilei/ACSE-universal-ansatze/fig/ fig1

backup_folder=/public/home/weileizeng/backup
backup:
	rsync -rP . ${backup_folder}/20250613/ --exclude env --exclude .git --exclude data --exclude outdated
#	rsync -rP . ../backup/ACSE-universal-ansatze/20250613/ --exclude env --exclude .git --exclude data --exclude outdated
test:
	$(py) test.py



## hubbard model
data-hubbard:
	$(py) data_generator_Hubbard.py
hubbard:
	$(py) model_hubbard.py
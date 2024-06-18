GPU=0

.PHONY: data train plot plot-all test
#python3.10 -m venv env
#source env/bin/activate
py	=env/bin/python

all:my-model_F

my-model_F:
	$(py) model_F.py \
--tag="v0.2.1-F-rnn" \
--batch_size=4256 \
--num_hidden_layers=6 \
--hidden_size=1024  \
--data_file_limit=-1 \
--learning_rate=0.0001 \
--gpu=7

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
	$(py) data-generator-Mqubit.py

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
	rsync -rP root@10.200.69.64:/root/weilei/ACSE-universal-ansatze/fig/ fig

backup_folder=/public/home/weileizeng/backup
backup:
	rsync -rP . ${backup_folder}/20250613/ --exclude env --exclude .git --exclude data --exclude outdated
#	rsync -rP . ../backup/ACSE-universal-ansatze/20250613/ --exclude env --exclude .git --exclude data --exclude outdated
test:
	$(py) test.py

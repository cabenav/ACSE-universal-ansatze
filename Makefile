GPU=0

.PHONY: data train plot plot-all test
#python3.10 -m venv env
#source env/bin/activate
py	=env/bin/python

all:my-model
my-model:
	$(py) model.py --tag="v5" --batch_size=128 --num_hidden_layers=6 --gpu=0

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

backup:
	rsync -rP . ../backup/ACSE-universal-ansatze/20250528/ --exclude env --exclude .git
test:
	$(py) test.py

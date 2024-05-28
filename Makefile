.PHONY: data train plot
#python3.10 -m venv env
#source env/bin/activate
py	=env/bin/python

all:model

data:
	$(py) data-generator-Mqubit.py
GPU=2
model:
	CUDA_VISIBLE_DEVICES=${GPU} $(py) model.py
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

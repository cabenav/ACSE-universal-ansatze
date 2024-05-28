.PHONY: data train plot
#python3.10 -m venv env
#source env/bin/activate
py	=env/bin/python

all:model

data:
	$(py) data-generator-Mqubit.py
GPU=5
model:
	CUDA_VISIBLE_DEVICES=${GPU} $(py) model.py

plot:
	$(py) plot.py ${filename_loss}

nvtop:
	nvidia-smi |head -n 15

#sync figs to weilei's macbook
rsync:
	rsync -rP root@10.200.69.64:/root/weilei/ACSE-universal-ansatze/fig/ fig

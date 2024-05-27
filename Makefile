#python3.10 -m venv env
#source env/bin/activate
py	=env/bin/python

all:train

GPU=5

train:
	CUDA_VISIBLE_DEVICES=${GPU} $(py) model.py
plot:
	$(py) plot.py

nvtop:
	nvidia-smi |head -n 15

#sync figs to weilei's macbook
rsync:
	rsync -rP root@10.200.69.64:/root/weilei/ACSE-universal-ansatze/fig/ fig

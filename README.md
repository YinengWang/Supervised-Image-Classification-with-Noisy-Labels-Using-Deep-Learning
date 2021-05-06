# Supervised-Image-Classification-with-Noisy-Labels-Using-Deep-Learning
A course project for DD2424: Deep Learning in Data Science at KTH

To setup a conda env:
```console
dd2424@server2:~$ conda update -n base conda
dd2424@server2:~$ conda create --name gcp-conda-cuda11-python37 python=3.7
dd2424@server2:~$ conda activate gcp-conda-cuda11-python37
(gcp-conda-cuda11-python37) dd2424@server2:~$ conda install pytorch torchvision cudatoolkit=11 -c pytorch
(gcp-conda-cuda11-python37) dd2424@server2:~$ conda install --file requirements.txt
```

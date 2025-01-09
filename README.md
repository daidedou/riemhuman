# A Riemannian Framework for Analysis of Human Body Surface

This repository contains the code for the paper "A Riemannian Framework for Analysis of Human Body Surface", published in WACV 2022.

### Installation
First install pytorch in you [preferred way](https://pytorch.org/get-started/previous-versions/), then run
```
pip install -r requirements.txt
```

### Interpolation
First download the [FAUST dataset](https://faust-leaderboard.is.tuebingen.mpg.de/), then run 
```
python polyscope_demo.py --faust_path path_where_you_downloaded_faust
```
The faust_path is the training/registrations/ folder of the dataset

### Distances 
To reproduce the figures of the paper :
```
python distance_demo.py --faust_path path_where_you_downloaded_faust
```
You can set '--type_exp pose' if you want to see results with the pose adapted metric

### Karcher Mean
```
python path_solver.py --faust_path path_where_you_downloaded_faust --karcher
```

In all scripts, you can set the a,b,c parameter as you wish. You can also try other energies like SRNF or [more complicated ones](https://hal.science/hal-01142780/document).

### Training
To compute the deformation basis, you'll need to download [DFAUST](https://dfaust.is.tue.mpg.de/), then run:
```
python basis_computation.py --dfaust_path path_where_dfaust --smpl_path path_where_faust
```
If you want to apply your code to your own **registered** data, you'll need to adapt the code accordingly

### Citing 

If you found our work useful, please cite us
```
@inproceedings{pierson2022riemannian,
  title={A riemannian framework for analysis of human body surface},
  author={Pierson, Emery and Daoudi, Mohamed and Tumpach, Alice-Barbara},
  booktitle={Proceedings of the IEEE/CVF winter conference on applications of computer vision},
  pages={2991--3000},
  year={2022}
}
```
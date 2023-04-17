# SePaint: Semantic Map Inpainting via Multinomial Diffusion

## TODO
Training and testing for V2X data

## Install Instructions

Create a conda environment by: `conda env create -f ./diff.yml`

In the folder containing `setup.py`, run
```
pip install --user -e .
```
The `--user` option ensures the library will only be installed for your user.
The `-e` option makes it possible to modify the library, and modifications will be loaded on the fly.

You should now be able to use it.

## Datasets
### Cityscapes
Download the data from https://www.cityscapes-dataset.com and run ``data_to_npy.py``.

### nuScenes
Download the processed nuScenes data [here](https://drive.google.com/file/d/1UnXLM4fZaGG2gy9IzGSS3FNJIY_9nESL/view?usp=share_link)

### V2X
TBD

## Training
Please specify the `LOG_PATH` in the following commands before the training. Your trained model will be saved there.

* Train for __cityscapes_coarse__: ```python train.py --eval_every 1 --check_every 25 --epochs 500 --log_home LOG_PATH --diffusion_steps 4000 --dataset cityscapes_coarse --batch_size 32 --dp_rate 0.1 --augmentation shift --lr 0.0001 --warmup 5 --batch_size 64```

  * Modify the ```is_inpa``` in ```cityscapes_fast.py``` as ```False```
* Train for __cityscapes_fine_large__: ```python train.py --eval_every 1 --check_every 25 --epochs 500 --log_home LOG_PATH --diffusion_steps 4000 --dataset cityscapes_coarse --batch_size 4 --dp_rate 0.1 --augmentation shift --lr 0.0001 --warmup 5```

  * Modify the ```is_inpa``` in ```cityscapes_fast.py``` as ```False```
* Train for __nuScenes__: ```python train.py --eval_every 1 --check_every 5 --epochs 60  --log_home LOG_PATH --diffusion_steps 4000 --dataset nuscenes --batch_size 16 --dp_rate 0.1 --augmentation shift --lr 0.0001 --warmup 5```
* Train for __V2X__: TBD

## Trained Models
We have provided trained models for three data [here](https://drive.google.com/file/d/1Nu3b4Ve_cfLYH0Tq8ntOzFnS9p5iYUga/view?usp=share_link): ```cityscapes_coarse```, ```cityscapes_fine_large```, and ```nuscenes```.

## Inpainting
Please specify the `MODEL_PATH_ROOT` in the following commands before the inpainting.

* `cd ./segmentation_diffusion`
* __Cityscapes__: ```python inpaint_sample.py --model MODEL_PATH_ROOT/cityscapes_fine_large```
  * Modify the ```is_inpa``` in ```cityscapes_fast.py``` as ```True```
* __nuScenes__: ```python inpaint_sample.py --model MODEL_PATH_ROOT/nuscenes```
  * Set ```is_extra_mask``` as False if you are using `unprojection` data. Otherwise set as `True`

## Acknowledgements
The code is based on [Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions](https://github.com/ehoogeboom/multinomial_diffusion)

## License
[MIT](./LICENSE.txt)
Convolutional Recurrent Neural Network
======================================

This software implements the Convolutional Recurrent Neural Network (CRNN) in paddlepaddle.
Origin software could be found in [crnn](https://github.com/bgshih/crnn)


## Requirements
* padddlepaddle 2.0+

## Data Preparation
Prepare a text in the following format
```
/path/to/img/img.jpg\tlabel
...
```

## Train
1. config the `train_list` and `eval_list` in [config.py](config.py)
2. generate alphabet
use fellow script to generate `dict.txt` in the some folder with `train.py` 
```sh
python3 utils/get_keys.py
```
2. use following script to run
```sh
python3 train.py
```

## Predict 
[infer.py](infer.py) is used to inference on single image

1. config `model_path`, `img_path` in [predict.py](src/scripts/predict.py)
2. use following script to predict
```sh
python3 predict.py
```
# EI-Seg

This repository contains the implementation of EI-Seg.
***

## Usage

### Set Data Path in conf.yaml

Set data path and dataset name in the config.yaml.
Also, all the hyperparameters is set in the config.yaml such as batch size and epochs.

### Train
Run the script below to train ei_seg in the normal way

```shell
python scripts/train.py config=ei_seg
```

Run the script below to train ei_seg in the cross-validation way

```shell
python scripts/train_cycleval.py config=ei_seg
```

### Predict

Run the script below to predict ei_seg in the normal way

```shell
python scripts/predict.py config=ei_seg
```

Run the script below to predict ei_seg in the cross-validation way

```shell
python scripts/predict_cycleval.py config=ei_seg
```

## Cross validation
It is noteworthy that we design an auto-record ckpt system in the condition of cross-validation training. For instance, when training ei_seg in the 5-fold cross-validtion way, the script will record the ckpt with which the model perform the best in the validation for each fold. There will be a 'ei-seg-5.csv' be generated in ./CrossValidationLogger. If it existes, the script will copy it into 'ei-seg-5-backup.csv' automatically. 

When predicting in the condition of cross-validation, there is a option of 'ckpt_from_csv'. If it is set as 'True', the script will autometically load the ckpt from the corresponding csv.


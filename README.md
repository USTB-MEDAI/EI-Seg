# Medical-Template

This is a template for medical image segmentation by Serein.
***

## Usage

### Set Data Path in conf.yml

all paramters are in `conf.yml`

### Train

```shell
torchrun --nproc_per_node 1 --master_port 12341 train.py --gpus 0 -o ./logs/norm_aug/unetr33/ --conf_path ./conf/unetr.yml --use_scheduler
```
`-o` : logs will output in this path
`--network` : choose which network
`--use_sheduler` : determine whether to use scheduler
`--patch_size` : patch size of input image
### Predict
```shell
c
```
torchrun --nproc_per_node 1 --master_port 12367 predict.py --gpus 0 -o ./prediction/norm_aug/IXI/unetr22/ --conf_path ./conf/unetr.yml -k ./logs/norm_aug/unetr22

torchrun --nproc_per_node 1 --master_port 12365 predict.py --gpus 1 -o ./prediction/2d3dloss_dice_dec_decouple17/ --conf_path ./conf/stdc2d.yml -k ./logs/2d3dloss_dice_dec_decouple17

### opentsne
torchrun --nproc_per_node 1 --master_port 12391 opentsne_train.py --gpus 0 --conf_path ./conf/opentsne.yml -k ./logs/opentsne/stdc_nodecouple -o ./tsne/figure12/test/stdc_nodecouple/

torchrun --nproc_per_node 1 --master_port 12390 opentsne_train.py --gpus 0 --conf_path ./conf/opentsne.yml -k ./logs/opentsne/stdc_midas -o ./tsne/figure12/test/stdc_midas/


torchrun --nproc_per_node 1 --master_port 12392 opentsne_train.py --gpus 0 --conf_path ./conf/opentsne.yml -k ./logs/opentsne/stdc_nodec_nodecouple -o ./tsne/figure12/test/stdc_nodec_nodecouple/

### our_no_decouple

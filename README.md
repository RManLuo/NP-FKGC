# NP-FKGC

## Requirement
```
pytorch==1.11
tqdm==4.64
normflows==1.4
dgl==0.9.0
tensorboardx==2.5.1
```

## Environment
* python 3.8
* Ubuntu 22.04
* RTX3090/A100
* Memory 32G/128G

## Dataset
* [NELL/WIKI](https://github.com/xwhan/One-shot-Relational-Learning)
* [FB15K](https://github.com/SongW-SW/REFORM)

## Train
NELL
```bash
python main.py --dataset NELL-One --data_path ./NELL --few 5 --data_form Pre-Train --prefix np_rgcn_attn_planar_nellone_5shot_intrain --device 0 --batch_size 128 --flow Planar --g_batch 1024
```

WIKI
```bash
python main.py --dataset Wiki-One --data_path ./wiki --few 5 --data_form Pre-Train --prefix np_rgcn_attn_planar_wiki_5shot_intrain_g_batch_1024_eval_8 --device 0 --batch_size 64 --flow Planar -dim 50 --g_batch 1024 --eval_batch 8 --eval_epoch 4000
```

FB15K
```bash
python main_gana.py --dataset FB15K-One --data_path ./FB15K --few 5 --data_form Pre-Train --prefix np_rgcn_attn_planar_fb15k_5shot_intrain --device 0 --batch_size 128 --flow Planar --g_batch 1024 --eval_batch_size 128
```
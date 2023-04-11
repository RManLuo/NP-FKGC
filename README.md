# NP-FKGC

## Requirement
```
pytorch==1.11
tqdm==4.64
normflows==1.4
dgl==0.9.0
tensorboardx==2.5.1
```
Note: Please make sure `dgl==0.9.0` and use CUDA, our codes rely on a small [bug](https://github.com/dmlc/dgl/issues/4512#issuecomment-1250642930) of dgl for running.

## Environment
* python 3.8
* Ubuntu 22.04
* RTX3090/A100
* Memory 32G/128G

## Dataset & Checkpoint
### Original Dataset
* [NELL/WIKI](https://github.com/xwhan/One-shot-Relational-Learning)
* [FB15K](https://github.com/SongW-SW/REFORM)
### Processed Dataset
* [Dataset](https://drive.google.com/drive/u/0/folders/1vN1AMapGZaUnQ4c7gPiBmO_nB6vvhj1c)
* [Checkpoint](https://drive.google.com/drive/u/0/folders/1gpHkQDgr5KzAXptl_fa1pATvk__prYUc)

Download the datasets and extract to the project root folder.  

## Train
NELL (3090)
```bash
python main.py --dataset NELL-One --data_path ./NELL --few 5 --data_form Pre-Train --prefix np_rgcn_attn_planar_nellone_5shot_intrain --device 0 --batch_size 128 --flow Planar --g_batch 1024
```

WIKI (A100)
```bash
python main.py --dataset Wiki-One --data_path ./Wiki --few 5 --data_form Pre-Train --prefix np_rgcn_attn_planar_wiki_5shot_intrain_g_batch_1024_eval_8 --device 0 --batch_size 64 --flow Planar -dim 50 --g_batch 1024 --eval_batch 8 --eval_epoch 4000
```

FB15K (3090)
```bash
python main.py --dataset FB15K-One --data_path ./FB15K --few 5 --data_form Pre-Train --prefix np_rgcn_attn_planar_fb15k_5shot_intrain --device 0 --batch_size 128 --flow Planar --g_batch 1024 --eval_batch_size 128 --K 14
```

## Eval
Download the checkpoint and extract to the `state/` folder.

NELL
```bash
python main.py --dataset NELL-One --data_path ./NELL --few 5 --data_form Pre-Train --prefix np_rgcn_attn_planar_nellone_5shot_intrain_0.46 --device 0 --batch_size 128 --flow Planar --g_batch 1024 --step test
```

WIKI
```bash
python main.py --dataset Wiki-One --data_path ./Wiki --few 5 --data_form Pre-Train --prefix np_rgcn_attn_planar_wiki_5shot_intrain_g_batch_1024_eval_8_0.503 --device 0 --batch_size 64 --flow Planar -dim 50 --g_batch 1024 --eval_batch 8 --eval_epoch 4000 --step test
```

FB15K
```bash
python main.py --dataset FB15K-One --data_path ./FB15K --few 5 --data_form Pre-Train --prefix np_rgcn_attn_planar_fb15k_5shot_intrain_0.536 --device 0 --batch_size 128 --flow Planar --g_batch 1024 --eval_batch_size 128 --K 14 --step test
```


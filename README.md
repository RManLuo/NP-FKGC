# NP-FKGC
Official code implementation for SIGIR 23 paper [Normalizing Flow-based Neural Process for Few-Shot Knowledge Graph Completion](https://arxiv.org/abs/2304.08183)
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
* [FB15K-237](https://github.com/SongW-SW/REFORM)
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

FB15K-237 (3090)
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

FB15K-237
```bash
python main.py --dataset FB15K-One --data_path ./FB15K --few 5 --data_form Pre-Train --prefix np_rgcn_attn_planar_fb15k_5shot_intrain_0.536 --device 0 --batch_size 128 --flow Planar --g_batch 1024 --eval_batch_size 128 --K 14 --step test
```

## Results
5-shot FKGC results
| Dataset | MRR   | Hits@10 | Hits@5 | Hits@1 |
| ------- | ----- | ------- | ------ | ------ |
| NELL    | 0.460 | 0.494   | 0.471  | 0.437  |
| WIKI    | 0.503 | 0.668   | 0.599  | 0.423  |
| FB15K-237   | 0.538 | 0.671   | 0.593  | 0.476  |

See full results in our paper.

## Citations
If you use this repo, please cite the following paper.
```
@inproceedings{
 luo2023npfkgc,
 title={Normalizing Flow-based Neural Process for Few-Shot Knowledge Graph Completion},
 author={Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan},
 booktitle={The 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
 year={2023}
}
```

## Acknowledgement
This repo is mainly based on [GANA](https://github.com/ngl567/GANA-FewShotKGC). We thank the authors for their great works.

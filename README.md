# IJCAI25-TUMFNet

### Path Setting
Run the following command to set paths:
```
cd <PATH>
python tracking/create_default_local_file.py --workspace_dir . --data_dir <PATH_of_Datasets> --save_dir ./output
```
You can also modify paths by these two files:
```
./lib/train/admin/local.py  # paths for training
./lib/test/evaluation/local.py  # paths for testing
```

### Training
Dowmload the pretrained [foundation model](https://www.kaggle.com/datasets/zhaodongding/drgbt603-results/data) (OSTrack and DropMae)
and put it under ./pretrained/.
```
CUDA_VISIBLE_DEVICES=0,1  NCCL_P2P_LEVEL=NVL nohup  python tracking/train.py --script drgbt --config DRGBT603 --save_dir ./output --mode multiple --nproc_per_node 1 >  train_track.log &
```
To enable the second-phase training, please set `second_phase` to `True` in `lib/train/actors/bat.py`.
```
out_dict = self.net(template=template_list,
                    search=search_img,
                    ce_template_mask=box_mask_z,
                    ce_keep_rate=ce_keep_rate,
                    return_last_attn=False,
                    second_phase=False,#is second phase
                    )
```

Modify the <DATASET_PATH> and <SAVE_PATH> in```./RGBT_workspace/test_rgbt_mgpus.py```, then run:
```
bash eval_drgbt.sh
```
In this way, you can obtain the experimental results and then run the following command to evaluate them:
```
python evaluate_DRGBT603\eval_DRGBT603.py
```

## Acknowledgment
- This repo is based on [BAT](https://github.com/SparkTempest/BAT) which is an exellent work, helps us to quickly implement our ideas.
- Thanks for the [OSTrack](https://github.com/botaoye/OSTrack) and [PyTracking](https://github.com/visionml/pytracking) library.

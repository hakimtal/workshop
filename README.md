## UW-Madison GI Tract Image Segmentation
## 3rd Solution

### Data Prepare
(1) Download the dataset to the seg_models/data  
Train data:  seg_models/data/train  
(2) Install the required packages
```shell
cd seg_models
pip install -r requirements.txt
```

### Detection Part 
Image data process, Map raw image pixels to 0 to 255 for training detection model

```shell
cd det_models
python normal_uw_images.py ../data/train/ ./datasets/images/
```
train detection model
```shell
cd det_models
sh train.sh
```
infer all images (include training images), then will get train_add_effdet0.json for for auxiliary training.

```shell
cd det_models
sh infer.sh
```
After doing the above, ./det_models/train_add_effdet0.json will be generated 

### Semantic Segmentation Part
A 3-slice model was removed from the solution, making the final score better than the current leaderboard,
use this 4x cls models and 4x seg model, it can get Private Score/Public Score =
0.88318/0.89023
#### preprocess
image data 5 folds split
```shell
cd seg_models
python k_folds_split.py
```
Creat 5 slice stack image and mask
```shell
cd seg_models
python preprocess.py
python preprocess_mask.py
```
After doing the above, we will get:  
(1) seg_models/data/train_stack_5_slice  
(2) seg_models/data/masks  
(3) seg_models/data/train_fold_updata.csv
#### train cls model
prtrained models with small size
```shell
cd seg_models
sh run_cls_pretrained.sh
```
train with large size
```shell
cd seg_models
sh run_cls_models.sh
```
Stochastic weight averaging with different snapshots
```shell
cd seg_models
sh run_cls_swa.sh
```

#### train seg model
train with large size(Use classification model as pretraining
)
```shell
cd seg_models
sh run_seg_models.sh
```
Stochastic weight averaging with different snapshots
```shell
cd seg_models
sh run_seg_swa.sh
```
Fine-tune the model at large scale (little effect on results)
```shell
cd seg_models
sh run_seg_models_finetune.sh
sh run_seg_swa_finetune.sh
```

#### predict
see predict.ipynb if want run predict on local. If you need to run this file, please make 
sure ./data/test and ./data/sample_submission.csv are not empty

```shell
cd seg_models
sh run_predict.sh
```

### Leaderboard:
- Public:  0.89162
- Private: 0.88255
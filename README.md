# GTP

Code for SIGIR-2026 Full Paper:  
**GTP: Mitigating Popularity Bias in PLM-based Sequential Recommendation via Group-Aware Token Pruning**

### STEP 0: Prepare the environment
* conda create -n gtp python=3.8
* conda activate gtp
* while read requirement; do pip install "$requirement"; done < requirements.txt (in Linux)
* install torch_scatter through .whl file from https://pytorch-geometric.com/whl/, such as:
  * wget https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
  * pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
  
### STEP 1: Prepare the dataset.
Please keep the following folder structure, download the dataset from the official URL in the paper, and download the pre-trained model from the Huggingface:
```python
-gtp.py
-run_gtp.py
...
-dataset
  -Metadata
    -meta_Office_Products.json.gz
  -Ratings
    -Office_Products.csv
-plm
  -opt-125m
```

### STEP 2: Preprocess the dataset.
* python preprocess_amazon.py --dataset=Office

### STEP 3: Pre-training the ranker model
* For multi GPUs:
  * nohup python run_gtp.py --dataset=Office --num_gpus=4 --batch_size=24 --distributed --multiGPU --valid_ratio=0.1 --train_stage=1 --thre 0.36 > Office_stage1.log 2>&1 &  
* For single GPU:
  * nohup python run_gtp.py --dataset=Office --batch_size=24 --valid_ratio=0.1 --train_stage=1 --thre 0.36> Office_stage1.log 2>&1 &

### STEP 4: Fine-tuning the ranker model
* For multi GPUs:
  * nohup python run_gtp.py --dataset=Office --num_gpus=4 --batch_size=24 --distributed --multiGPU --valid_ratio=0.1 --train_stage=2 > Office_stage2.log 2>&1 &  
* For single GPU:
  * nohup python run_gtp.py --dataset=Office --batch_size=24 --valid_ratio=0.1 --train_stage=2 > Office_stage2.log 2>&1 &  

### STEP 5: Self-distillation
* For multi GPUs:
  * nohup python run_gtp.py --dataset=Office --num_gpus=4 --batch_size=246 --distributed --multiGPU --valid_ratio=0.1 --train_stage=3 > Office_stage3.log 2>&1 &  
* For single GPU:
  * nohup python run_gtp.py --dataset=Office --batch_size=24 --valid_ratio=0.1 --train_stage=3 > Office_stage3.log 2>&1 &  

### Notes:
* In our experiments, we use 4 gpus and the batch size on each gpu is 24. Thus the total batch size is 96.
* The batch size on each gpu is important in stage 3 because the value of sampled cross-entropy loss is related to the negatives, and we sample (batch_size * 10) negative items for each batch. Changing the batch size may not get the ideal output.
* We have tested the codes for a quick start and reproducing the results reported in the paper. If you have any questions and find any bug, please let us know in the review, and we will fix the bug in the next version
* If you want to use OPT-125M directly from Hugging Face instead of downloading it locally, simply set the root_path to an empty string ("").

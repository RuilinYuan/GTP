import os
import pickle
import torch.multiprocessing as mp
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils import LossMeter
from trainer_base import TrainerBase
from param import parse_args
import time
from torch.cuda.amp import GradScaler
from torch import autocast
from metrics import cal_recall, cal_ndcg, cal_gini, cal_cratio
from tqdm import tqdm
from utils import info
import socket
import random
import numpy as np
from dataloader import get_dataloader
from gtp import STAPLE


# The Trainer inherits TrainerBase in trainer_base.py
class HypernetTrainer(TrainerBase):
    def __init__(self, args, tokenizer, train_loader=None, val_loader=None, test_loader=None, train=False):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        self.model = STAPLE(args, tokenizer)
        self.tokenizer = tokenizer

        # GPU Options
        info(f'Model Launching at GPU {self.args.gpu}')
        self.model = self.model.to(args.gpu)

        # Load model weights based on training stage
        if args.load:
            self.load(args.load)

        self.loss_names = ['total_loss', 'rec_loss', 'rec_debias']
        self.best_valid_result = 0
        self.early_stop_step = 0

        # Print trainable parameters
        self.print_trainable_parameters()

    @torch.no_grad()
    def valid_epoch(self, epoch, mode='valid'):
        dataloader = self.val_loader if mode == 'valid' else self.test_loader
        self.model.eval()
        info(f"GPU memory before test: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        with autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.fp16):
            if self.args.distributed:
                self.model.module.generate_embs(dataloader.dataset.get_items_tokens())
                dist.barrier()
            else:
                self.model.generate_embs(dataloader.dataset.get_items_tokens())
        loader_length = len(dataloader)
        logger_batch = (loader_length // 10) + 1
        predict_score = []
        label = []
        example_index = []
        candidate_items = []
        real_label = []

        # Add tqdm progress bar
        for batch_idx, batch_data in tqdm(enumerate(dataloader), total=loader_length, desc="Testing",
                                          disable=False):
            self.transfer_device(batch_data)
            if (batch_idx % logger_batch) == 0:
                info(f"GPU {self.args.gpu}-Evaluation: {batch_idx}/{loader_length}")
            with autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.fp16):
                if self.args.distributed:
                    scores, bs_label = self.model.module.valid_step(batch_data)
                else:# bs_label就是目标物品
                    scores, bs_label = self.model.valid_step(batch_data)
            example_index.append(batch_data['example_index'].cpu())
            label.append(bs_label.cpu())
            predict_score.append(scores.cpu())
            candidate_items.append(batch_data['negative_items'].cpu())
            real_label.append(batch_data['target_iid'].cpu())


        example_index = torch.cat([x.cpu() for x in example_index], dim=0)
        predict_score = torch.cat([x.cpu() for x in predict_score], dim=0).to(example_index.device)
        label = torch.cat([x.cpu() for x in label], dim=0).to(example_index.device)
        candidate_items = torch.cat([x.cpu() for x in candidate_items], dim=0).to(example_index.device)
        real_label = torch.cat([x.cpu() for x in real_label], dim=0).to(example_index.device)

        if self.args.distributed:
            all_predict_score = [torch.zeros_like(predict_score) for _ in range(self.args.num_gpus)]
            dist.all_gather(all_predict_score, predict_score.contiguous())

            all_label = [torch.zeros_like(label) for _ in range(self.args.num_gpus)]
            dist.all_gather(all_label, label.contiguous())

            all_example_index = [torch.zeros_like(example_index) for _ in range(self.args.num_gpus)]
            dist.all_gather(all_example_index, example_index.contiguous())

            all_candidate_items = [torch.zeros_like(candidate_items) for _ in range(self.args.num_gpus)]
            dist.all_gather(all_candidate_items, candidate_items.contiguous())

            all_real_label = [torch.zeros_like(real_label) for _ in range(self.args.num_gpus)]
            dist.all_gather(all_real_label, real_label.contiguous())

            predict_score, label = self.clean_dist_duplicate(all_predict_score, all_label, all_example_index)
            candidate_items, _ = self.clean_dist_duplicate(all_candidate_items, all_label, all_example_index)
            real_label, _ = self.clean_dist_duplicate(all_real_label, all_label, all_example_index)

        recall = cal_recall(label.cpu(), predict_score.cpu(), [10, 50, 100, 200])
        ndcg = cal_ndcg(label.cpu(), predict_score.cpu(), [10, 50, 100, 200])
        gini = cal_gini(predict_score.cpu(), self.args.item_count, [10, 50, 100, 200])
        cratio = cal_cratio(predict_score.cpu(), self.args.item2pop, [10, 50, 100, 200])

        info(f"\nRecall:{recall}\nNDCG:{ndcg}\nGini:{gini}\nCRatio:{cratio}")

        return {'save': False, 'exit': True, 'result': [ndcg[-1]]}


    def clean_dist_duplicate(self, all_predict_score, all_label, all_example_index):
        all_predict_score = torch.concat(all_predict_score, dim=0).cpu()
        all_label = torch.concat(all_label, dim=0).cpu()

        predict_score = torch.zeros_like(all_predict_score)
        label = torch.zeros_like(all_label)
        example_index = torch.concat(all_example_index, dim=0).cpu()

        predict_score[example_index] = all_predict_score
        label[example_index] = all_label
        exp_cnt = max(example_index) + 1

        return predict_score[:exp_cnt], label[:exp_cnt]

    def transfer_device(self, data):
        device = next(self.model.parameters()).device
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)

    def save(self, path):
        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
        saved_parameters = {}
        model_generator = self.model.named_parameters() if not self.args.distributed else self.model.module.named_parameters()
        for param_name, param in model_generator:
            if param.requires_grad:
                saved_parameters[param_name] = param
        torch.save(saved_parameters, path)

    def load(self, path, loc=None):
        weights = torch.load(path, map_location=next(self.model.parameters()).device)
        if self.args.distributed:
            info(self.model.module.load_state_dict(weights, strict=False))
        else:
            info(self.model.load_state_dict(weights, strict=False))

    def save_pickle(self, obj, path):
        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
        pickle.dump(obj, open(path, 'wb'))

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        model_generator = self.model.named_parameters() if not self.args.distributed else self.model.module.named_parameters()
        for _, param in model_generator:
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        info(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )


    def test(self):
        """Run test-only mode."""
        info("Running test mode")
        if self.args.distributed:
            dist.barrier()
        self.valid_epoch(-1, mode='test')
        if self.args.distributed:
            dist.barrier()

        info("Test finished!")


def main_worker(gpu, args):
    args.rank = gpu
    info(f'Process Launching at GPU {args.gpu}')

    if args.distributed:
        random.seed(args.seed + args.rank)
        np.random.seed(args.seed + args.rank)
        torch.manual_seed(args.seed + args.rank)
        torch.cuda.manual_seed(args.seed + args.rank)

        torch.cuda.set_device(args.gpu)
        args.world_size = args.num_gpus
        args.dist_backend = "nccl"
        args.dist_url = f'tcp://127.0.0.1:{args.port}'
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    info(f'Building test loader at GPU {args.gpu}')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.root_path + args.backbone)

    # Only load test_loader
    _, _,test_loader  = get_dataloader(args, tokenizer)

    # Define trainer with test_loader only
    trainer = HypernetTrainer(args, tokenizer, test_loader=test_loader, train=False)
    trainer.test()


if __name__ == "__main__":
    args = parse_args()
    args.test_only = True
    args.distributed = False
    args.valid_first = False
    args.fp16 = True  # Enable mixed precision

    # 根据训练阶段加载对应模型权重
    if args.train_stage == 1:
        args.load = args.output + args.dataset + '-1.pth'
    elif args.train_stage == 2:
        args.load = args.output + args.dataset + '-2.pth'
    elif args.train_stage == 3:
        args.load = args.output + args.dataset + '-3.pth'

    info("============runner run with args=================")
    info(args)
    main_worker(args.gpu, args)

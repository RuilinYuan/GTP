import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch import autocast
from utils import info
from modules.layers import MultiLayerPerceptron, LBSign

class STAPLE(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args

        from transformers import OPTModel
        self.llm = OPTModel.from_pretrained(args.root_path + args.backbone)
        self.llm_debias = OPTModel.from_pretrained(args.root_path + args.debias_backbone)

        self.en_de_bias_debias = nn.Sequential(
            nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size),
            nn.LeakyReLU(),
        )

        self.en_de_bias = nn.Sequential(
            nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size),
            nn.LeakyReLU(),
        )

        self.hypernet = MultiLayerPerceptron(
            self.llm.config.hidden_size + 128,
            [1024, 512, 256],
            output_layer=False,
            dropout=0.0,
            use_bn=True
        )
        self.mask_layer = nn.Linear(256, 1)

        self.sign = LBSign.apply
        self.thre = args.thre

        self.tfidf_expander = nn.Linear(1, 64)
        self.pop_expander = nn.Linear(2, 64)

        self._popular_unique_tokens = set()
        self._tail_unique_tokens = set()
        self._shared_tokens = set()

        self._token_pop_ratios = {}
        self._token_tail_ratios = {}
        self._token_distribution_computed = False

        self.pop_output_counter=0

        self.freeze_stage_params()
        self.item_embs = None
        self.item_embs_debias = None

    def freeze_stage_params(self):
        if self.args.train_stage == 1:
            for param in self.llm.parameters():
                param.requires_grad = True
            for param in self.en_de_bias.parameters():
                param.requires_grad = True
            for param in self.hypernet.parameters():  # Student hypernet
                param.requires_grad = True
            for param in self.mask_layer.parameters():  # Student mask layer
                param.requires_grad = True
            # Shared linear layers
            for param in self.tfidf_expander.parameters():
                param.requires_grad = True
            for param in self.pop_expander.parameters():
                param.requires_grad = True


        if self.args.train_stage == 2:
            if not os.path.isfile(self.args.output + f"{self.args.dataset}-1.pth"):
                raise NotImplementedError('Missing stage1 checkpoint!')
            weights_stage1 = torch.load(self.args.output + f"{self.args.dataset}-1.pth",
                                        map_location=next(self.llm_debias.parameters()).device)
            info(self.load_state_dict(weights_stage1, strict=False))
            ## 冻结PLM模型参数，只更新MLP
            for param in self.llm.parameters():
                param.requires_grad = False
            for param in self.en_de_bias.parameters():
                param.requires_grad = True
            for param in self.hypernet.parameters():  # Student hypernet
                param.requires_grad = False
            for param in self.mask_layer.parameters():  # Student mask layer
                param.requires_grad = False
            # Shared linear layers
            for param in self.tfidf_expander.parameters():
                param.requires_grad = False
            for param in self.pop_expander.parameters():
                param.requires_grad = False

        if self.args.train_stage == 3:
            if not os.path.isfile(self.args.output + f"{self.args.dataset}-1.pth"):
                raise NotImplementedError('Missing stage1 checkpoint!')
            if not os.path.isfile(self.args.output + f"{self.args.dataset}-2.pth"):
                raise NotImplementedError('Missing stage2 checkpoint!')
            weights_stage1 = torch.load(self.args.output + f"{self.args.dataset}-1.pth",
                                        map_location=next(self.llm_debias.parameters()).device)
            weights_stage2 = torch.load(self.args.output + f"{self.args.dataset}-2.pth",
                                        map_location=next(self.llm_debias.parameters()).device)
            weights_stage1.update(weights_stage2)
            weights = weights_stage1

            teacher_weights = {}
            for key in weights.keys():
                if 'llm' in key:
                    teacher_weights[key.replace('llm', 'llm_debias')] = weights[key]
                if 'en_de_bias' in key:
                    teacher_weights[key.replace('en_de_bias', 'en_de_bias_debias')] = weights[key]

            student_hypernet_weights = {}
            for key in weights.keys():
                if not ('llm' in key or 'en_de_bias' in key):
                    student_hypernet_weights[key] = weights[key]

            info("Loading student hypernet weights:")
            for key in student_hypernet_weights.keys():
                info(f"  - {key}")

            info(self.load_state_dict(teacher_weights, strict=False))

            info(self.load_state_dict(student_hypernet_weights, strict=False))

            for param in self.llm.parameters():
                param.requires_grad = True
            for param in self.en_de_bias.parameters():
                param.requires_grad = True

            for param in self.llm_debias.parameters():
                param.requires_grad = False
            for param in self.en_de_bias_debias.parameters():
                param.requires_grad = False

            for param in self.hypernet.parameters():
                param.requires_grad = False
            for param in self.mask_layer.parameters():
                param.requires_grad = False

            for param in self.tfidf_expander.parameters():
                param.requires_grad = False
            for param in self.pop_expander.parameters():
                param.requires_grad = False

        for param in self.llm_debias.parameters():
            param.requires_grad = False

    def trainable2float(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                info(f"Trainable Parameter:{name}")
                param.data = param.data.float()

    def generate_mask(self, input_ids, tfidf_scores, pop_scores, tail_scores, use_debias=False):
        if use_debias:
            token_embeds = self.llm_debias.get_input_embeddings()(
                input_ids)  # Shape: (batch_size, seq_len, hidden_size)
        else:
            token_embeds = self.llm.get_input_embeddings()(input_ids)  # Shape: (batch_size, seq_len, hidden_size)

        tfidf_scores = tfidf_scores.unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
        pop_features = torch.stack([pop_scores, tail_scores], dim=-1)

        batch_size, seq_len, _ = tfidf_scores.shape
        tfidf_expanded = self.tfidf_expander(tfidf_scores)  # Shape: (batch_size, seq_len, 64)
        pop_expanded = self.pop_expander(pop_features)  # Shape: (batch_size, seq_len, 64）

        token_features = torch.cat([token_embeds, tfidf_expanded, pop_expanded],
                                    dim=-1)  # Shape: (batch_size, seq_len, hidden_size + 128)
        batch_size, seq_len, feature_dim = token_features.shape
        token_features_flat = token_features.view(-1, feature_dim)
        hypernet_training = self.hypernet.training
        if not self.training:
            self.hypernet.train()
        elif self.args.train_stage == 3 and self.args.bn:
            self.hypernet.eval()
        hyper_output = self.hypernet(token_features_flat)  # Shape: (batch_size * seq_len, 256)
        if not self.training:
            self.hypernet.eval()
        hyper_output = hyper_output.view(batch_size, seq_len, -1)
        mask_scores = self.mask_layer(hyper_output)  # Shape: (batch_size, seq_len, 1)
        mask_scores = torch.sigmoid(mask_scores)
        mask = self.sign(torch.relu(mask_scores - self.thre))  # Binary mask: 0 or 1

        return mask.squeeze(-1),mask_scores.squeeze(-1) # Shape: (batch_size, seq_len)
    def apply_mask(self,input_ids,attention_mask, mask):
        masked_attention_mask = attention_mask * mask  # 结合两种mask
        return masked_attention_mask

    def get_embedding(self, input_ids, attention_mask, tfidf_scores, pop_scores, tail_scores):
        """Get embedding with token masking."""
        mask, mask_scores = self.generate_mask(input_ids, tfidf_scores, pop_scores, tail_scores, use_debias=False)
        masked_attention_mask = self.apply_mask(input_ids,attention_mask, mask)
        llm_output = self.llm(input_ids=input_ids, attention_mask=masked_attention_mask)
        batch_size, seq_len = masked_attention_mask.shape
        position_indices = torch.arange(seq_len, device=masked_attention_mask.device).expand(batch_size, -1)
        masked_positions = position_indices * masked_attention_mask - (1 - masked_attention_mask)
        last_position = masked_positions.max(dim=1).values
        last_position = torch.clamp(last_position, min=0)
        embedding = self.gather_indexes(llm_output.last_hidden_state, last_position)
        embedding = embedding.float()
        if self.args.train_stage != 3:
            return self.en_de_bias(embedding), masked_attention_mask, mask_scores
        else:
            return embedding, masked_attention_mask, mask_scores


    def compute_pop_bias_loss(self, seq_mask, item_mask, seq_pop_scores, item_pop_scores, item_is_popular,
                              seq_attention_mask=None, item_attention_mask=None, seq_is_popular=None,
                              seq_item_positions=None, item_input_ids=None, seq_mask_scores=None,
                              item_mask_scores=None):

        import time
        start_time = time.time()
        device = item_mask.device
        unique_token_loss = torch.tensor(0.0, device=device, requires_grad=True)
        shared_token_loss = torch.tensor(0.0, device=device, requires_grad=True)


        popular_unique_scores = []
        tail_unique_scores = []
        shared_scores = []

        popular_unique_scores_sum = torch.tensor(0.0, device=device, requires_grad=True)
        popular_unique_scores_total = torch.tensor(0.0, device=device, requires_grad=True)
        tail_unique_scores_sum = torch.tensor(0.0, device=device, requires_grad=True)
        tail_unique_scores_total = torch.tensor(0.0, device=device, requires_grad=True)
        shared_kept_token = torch.tensor(0.0, device=device, requires_grad=True)
        shared_total = torch.tensor(0.0, device=device, requires_grad=True)

        if not hasattr(self, '_popular_unique_tensor'):
            self._popular_unique_tensor = torch.tensor(list(self._popular_unique_tokens),
                                                       device=device) if self._popular_unique_tokens else torch.tensor(
                [], device=device, dtype=torch.long)
            self._tail_unique_tensor = torch.tensor(list(self._tail_unique_tokens),
                                                    device=device) if self._tail_unique_tokens else torch.tensor([],
                                                                                                                 device=device,
                                                                                                                 dtype=torch.long)
            self._shared_tensor = torch.tensor(list(self._shared_tokens),
                                               device=device) if self._shared_tokens else torch.tensor([],
                                                                                                       device=device,
                                                                                                       dtype=torch.long)

        if item_mask is not None and item_is_popular is not None and item_attention_mask is not None and item_input_ids is not None:
            valid_positions = item_attention_mask > 0  # (batch_size, seq_len) 有效位置
            item_mask_filtered = item_mask * valid_positions.float()  # (batch_size, seq_len) dropout过后的有效位置
            popular_items = item_is_popular == 1  # (batch_size,)
            tail_items = item_is_popular == 0  # (batch_size,)

            popular_unique_kept_sum = torch.tensor(0.0, device=device, requires_grad=True)
            popular_unique_total_sum = torch.tensor(0.0, device=device, requires_grad=True)
            tail_unique_kept_sum = torch.tensor(0.0, device=device, requires_grad=True)
            tail_unique_total_sum = torch.tensor(0.0, device=device, requires_grad=True)

            if popular_items.any():
                popular_mask_values = item_mask_filtered[popular_items]  # (num_popular_items, seq_len) 流行物品的token保留mask情况
                popular_valid_positions = valid_positions[popular_items]  # (num_popular_items, seq_len)
                popular_item_ids = item_input_ids[popular_items]  # (num_popular_items, seq_len)

                if item_mask_scores is not None:
                    popular_mask_scores = item_mask_scores[popular_items]  # (num_popular_items, seq_len)
                else:
                    popular_mask_scores = popular_mask_values  # 使用mask值作为scores

                if hasattr(self, '_token_distribution_computed') and self._token_distribution_computed:
                    all_valid_mask_values = popular_mask_values[popular_valid_positions]
                    all_valid_mask_scores = popular_mask_scores[popular_valid_positions]
                    all_valid_token_ids = popular_item_ids[popular_valid_positions]

                    if all_valid_mask_values.numel() > 0:

                        is_popular_unique = (all_valid_token_ids.unsqueeze(1) == self._popular_unique_tensor).any(
                            dim=1).float()
                        is_shared = (all_valid_token_ids.unsqueeze(1) == self._shared_tensor).any(dim=1).float()

                        if is_popular_unique.any():
                            unique_mask_values = all_valid_mask_values[is_popular_unique == 1]
                            unique_mask_scores = all_valid_mask_scores[is_popular_unique == 1]
                            popular_unique_scores.append(unique_mask_scores)  # 收集独有token scores
                            popular_unique_kept_sum = popular_unique_kept_sum + unique_mask_values.sum()
                            popular_unique_total_sum = popular_unique_total_sum + is_popular_unique.sum()
                            popular_unique_scores_sum = popular_unique_scores_sum + unique_mask_scores.sum()
                            popular_unique_scores_total = popular_unique_scores_total + is_popular_unique.sum()

                        if is_shared.any():
                            shared_mask_values = all_valid_mask_values[is_shared == 1]
                            shared_mask_scores = all_valid_mask_scores[is_shared == 1]
                            shared_scores.append(shared_mask_scores)  # 收集共享token scores
                            shared_kept_token = shared_kept_token + shared_mask_values.sum()
                            shared_total = shared_total + is_shared.sum()


            if tail_items.any():
                tail_mask_values = item_mask_filtered[tail_items]  # (num_tail_items, seq_len)
                tail_valid_positions = valid_positions[tail_items]  # (num_tail_items, seq_len)
                tail_item_ids = item_input_ids[tail_items]  # (num_tail_items, seq_len)

                if item_mask_scores is not None:
                    tail_mask_scores = item_mask_scores[tail_items]  # (num_tail_items, seq_len)
                else:
                    tail_mask_scores = tail_mask_values  # 使用mask值作为scores

                if hasattr(self, '_token_distribution_computed') and self._token_distribution_computed:
                    all_valid_mask_values = tail_mask_values[tail_valid_positions]
                    all_valid_mask_scores = tail_mask_scores[tail_valid_positions]
                    all_valid_token_ids = tail_item_ids[tail_valid_positions]

                    if all_valid_mask_values.numel() > 0:
                        is_tail_unique = (all_valid_token_ids.unsqueeze(1) == self._tail_unique_tensor).any(
                            dim=1).float()
                        is_shared = (all_valid_token_ids.unsqueeze(1) == self._shared_tensor).any(dim=1).float()

                        if is_tail_unique.any():
                            unique_mask_values = all_valid_mask_values[is_tail_unique == 1]
                            unique_mask_scores = all_valid_mask_scores[is_tail_unique == 1]
                            tail_unique_scores.append(unique_mask_scores)  # 收集独有token scores
                            tail_unique_kept_sum = tail_unique_kept_sum + unique_mask_values.sum()
                            tail_unique_total_sum = tail_unique_total_sum + is_tail_unique.sum()
                            tail_unique_scores_sum = tail_unique_scores_sum + unique_mask_scores.sum()
                            tail_unique_scores_total = tail_unique_scores_total + is_tail_unique.sum()

                        if is_shared.any():
                            shared_mask_values = all_valid_mask_values[is_shared == 1]
                            shared_mask_scores = all_valid_mask_scores[is_shared == 1]
                            shared_scores.append(shared_mask_scores)  # 收集共享token scores
                            shared_kept_token = shared_kept_token + shared_mask_values.sum()
                            shared_total = shared_total + is_shared.sum()

        if popular_unique_scores_total > 0 and tail_unique_scores_total > 0:
            popular_unique_mean_score = popular_unique_scores_sum / (popular_unique_scores_total + 1e-8)
            tail_unique_mean_score = tail_unique_scores_sum / (tail_unique_scores_total + 1e-8)
            unique_token_loss = torch.relu(popular_unique_mean_score - tail_unique_mean_score)
        else:
            unique_token_loss = torch.tensor(0.0, device=device, requires_grad=True)

        shared_token_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if shared_scores and shared_total > 0:
            shared_combined = torch.cat(shared_scores)
            shared_token_loss = 1.0 - torch.mean(shared_combined)


        if not self.training or (hasattr(self.args, 'gpu') and self.args.gpu == 0):
            self.pop_output_counter += 1
            if self.pop_output_counter % 80 == 0:  # 每80个step输出一次详细统计

                if popular_unique_total_sum > 0:
                    popular_unique_ratio = popular_unique_kept_sum.item() / popular_unique_total_sum.item()
                    print(
                        f"Item Popular Unique Tokens - Kept: {popular_unique_kept_sum.item():.0f}/{popular_unique_total_sum.item():.0f} ({popular_unique_ratio:.4f})")
                if tail_unique_total_sum > 0:
                    tail_unique_ratio = tail_unique_kept_sum.item() / tail_unique_total_sum.item()
                    print(
                        f"Item Tail Unique Tokens - Kept: {tail_unique_kept_sum.item():.0f}/{tail_unique_total_sum.item():.0f} ({tail_unique_ratio:.4f})")
                # 添加共享token的统计信息
                if shared_total > 0:
                    shared_ratio = shared_kept_token.item() / shared_total.item()
                    print(
                        f"Item Shared Tokens - Kept: {shared_kept_token.item():.0f}/{shared_total.item():.0f} ({shared_ratio:.4f})")

                # 输出独有token的mask scores统计
                if popular_unique_scores:
                    popular_unique_scores_combined = torch.cat(popular_unique_scores)
                    print(f"Popular Unique Token Scores - Min: {popular_unique_scores_combined.min().item():.4f}, "
                          f"Max: {popular_unique_scores_combined.max().item():.4f}, "
                          f"Mean: {popular_unique_scores_combined.mean().item():.4f}")

                if tail_unique_scores:
                    tail_unique_scores_combined = torch.cat(tail_unique_scores)
                    print(f"Tail Unique Token Scores - Min: {tail_unique_scores_combined.min().item():.4f}, "
                          f"Max: {tail_unique_scores_combined.max().item():.4f}, "
                          f"Mean: {tail_unique_scores_combined.mean().item():.4f}")

                # 输出共享token的mask scores统计
                if shared_scores:
                    shared_scores_combined = torch.cat(shared_scores)
                    print(f"Shared Token Scores - Min: {shared_scores_combined.min().item():.4f}, "
                          f"Max: {shared_scores_combined.max().item():.4f}, "
                          f"Mean: {shared_scores_combined.mean().item():.4f}")

                # 输出层级约束损失信息
                if shared_token_loss > 0:
                    print(f"shared_token_loss: {self.args.shared_token_weight * shared_token_loss.item():.4f}")
                elapsed_time = time.time() - start_time
                # 输出执行时间
                print(f"compute_pop_bias_loss execution time: {elapsed_time * 1000:.2f} ms")
                print()

        return unique_token_loss,shared_token_loss

    def get_debias_embedding(self, input_ids, attention_mask, tfidf_scores, pop_scores, tail_scores):
        """Get debias embedding with token masking."""
        mask, mask_scores = self.generate_mask(input_ids, tfidf_scores, pop_scores, tail_scores, use_debias=True)
        masked_attention_mask = self.apply_mask(input_ids, attention_mask, mask)
        llm_output = self.llm_debias(input_ids=input_ids, attention_mask=masked_attention_mask)
        batch_size, seq_len = masked_attention_mask.shape
        position_indices = torch.arange(seq_len, device=masked_attention_mask.device).expand(batch_size, -1)
        masked_positions = position_indices * masked_attention_mask - (1 - masked_attention_mask)
        last_position = masked_positions.max(dim=1).values
        last_position = torch.clamp(last_position, min=0)
        embedding = self.gather_indexes(llm_output.last_hidden_state, last_position)
        embedding = embedding.float()
        return self.en_de_bias_debias(embedding), masked_attention_mask, mask_scores

    def reshape_item_cls(self, item_cls, negative_items):
        item_cls = item_cls.view(-1, item_cls.size()[-1])
        if self.args.nega_strategy == 'random':
            item_cls = item_cls.view(-1, self.args.train_nega_count + 1, item_cls.size()[-1])
            item_target_cls = item_cls[:, 0].unsqueeze(1)
            item_negative_cls = item_cls[:, 1:].reshape(1, -1, item_target_cls.size(2)).repeat(item_target_cls.size(0), 1, 1)
            item_cls = torch.cat([item_target_cls, item_negative_cls], dim=1)
            target_position = torch.zeros([item_cls.size(0)], device=item_cls.device).long()
            negative_items_target = negative_items[:, 0].unsqueeze(1)
            negative_items_others = negative_items[:, 1:].reshape(1, -1).repeat(item_target_cls.size(0), 1)
            negative_items = torch.cat([negative_items_target, negative_items_others], dim=1)
        elif self.args.nega_strategy == 'random+inbatch':
            batch_size = item_cls.size(0) // (self.args.train_nega_count + 1)
            item_cls = item_cls.unsqueeze(0).repeat(batch_size, 1, 1)
            target_position = torch.arange(
                item_cls.size(0),
                device=item_cls.device,
                dtype=torch.long
            ) * (self.args.train_nega_count + 1)
            negative_items = negative_items.reshape(1, -1).repeat(batch_size, 1)
        else:
            raise NotImplementedError
        return item_cls, target_position, negative_items

    def forward(self, inputs):
        seq_output = self.get_embedding(
            input_ids=inputs['sequence_input_ids'],
            attention_mask=inputs['sequence_attention_mask'],
            tfidf_scores=inputs['sequence_tfidf_scores'],
            pop_scores=inputs['sequence_pop_scores'],
            tail_scores=inputs['sequence_tail_scores']
        )

        seq_cls, seq_mask, seq_mask_scores = seq_output
        item_output = self.get_embedding(
            input_ids=inputs['item_input_ids'],
            attention_mask=inputs['item_attention_mask'],
            tfidf_scores=inputs['item_tfidf_scores'],
            pop_scores=inputs['item_pop_scores'],
            tail_scores=inputs['item_tail_scores']
        )
        item_cls, item_mask, item_mask_scores = item_output
        item_cls, target_position, negative_items = self.reshape_item_cls(item_cls, inputs['negative_items'])

        if self.args.train_stage == 3:
            with torch.no_grad():
                item_cls_teacher_output = self.get_debias_embedding(
                    input_ids=inputs['item_input_ids'],
                    attention_mask=inputs['item_attention_mask'],
                    tfidf_scores=inputs['item_tfidf_scores'],
                    pop_scores=inputs['item_pop_scores'],
                    tail_scores=inputs['item_tail_scores']
                )
                seq_cls_teacher_output = self.get_debias_embedding(
                    input_ids=inputs['sequence_input_ids'],
                    attention_mask=inputs['sequence_attention_mask'],
                    tfidf_scores=inputs['sequence_tfidf_scores'],
                    pop_scores=inputs['sequence_pop_scores'],
                    tail_scores=inputs['sequence_tail_scores']
                )
                item_cls_teacher, item_teacher_mask, item_teacher_mask_scores = item_cls_teacher_output
                seq_cls_teacher, seq_teacher_mask, seq_teacher_mask_scores = seq_cls_teacher_output

            item_cls_teacher, _, _ = self.reshape_item_cls(item_cls_teacher, inputs['negative_items'])
            item_cls_teacher = item_cls_teacher.float()
            seq_cls_teacher = seq_cls_teacher.float().unsqueeze(-1)
            scores_teacher = torch.bmm(item_cls_teacher, seq_cls_teacher).squeeze(-1)
        else:
            scores_teacher = None

        item_cls = item_cls.float()
        seq_cls = seq_cls.float().unsqueeze(-1)
        if self.args.scaled_dot:
            scores_student = torch.bmm(item_cls, seq_cls).squeeze(-1) / math.sqrt(item_cls.size()[-1])
        else:
            scores_student = torch.bmm(item_cls, seq_cls).squeeze(-1)

        rec_loss = F.cross_entropy(scores_student, target_position)

        if self.args.train_stage == 1:
            unique_token_loss, shared_token_loss = self.compute_pop_bias_loss(
                seq_mask=seq_mask,
                item_mask=item_mask,
                seq_pop_scores=inputs['sequence_pop_scores'],
                item_pop_scores=inputs['item_pop_scores'],
                item_is_popular=inputs['item_is_popular'],
                seq_attention_mask=inputs['sequence_attention_mask'],
                item_attention_mask=inputs['item_attention_mask'],
                seq_is_popular=inputs['seq_is_popular'],  # 添加用户历史物品的流行度信息
                seq_item_positions=inputs['seq_item_positions'],  # 添加用户历史物品的位置信息
                item_input_ids=inputs['item_input_ids'],  # 添加物品ID用于独有token统计
                seq_mask_scores=seq_mask_scores,
                item_mask_scores=item_mask_scores
            )
        else:
            unique_token_loss = torch.tensor(0.0, device=seq_cls.device, requires_grad=True)
            shared_token_loss = torch.tensor(0.0, device=seq_cls.device, requires_grad=True)

        if self.args.train_stage == 3 and self.args.debias_alpha != 0:
            rec_loss_debias = self.cal_debias_loss(scores_student, scores_teacher, target_position, negative_items)
        else:
            rec_loss_debias = torch.zeros_like(rec_loss)

        if self.args.train_stage == 1:
            total_loss = rec_loss + rec_loss_debias * self.args.debias_alpha + unique_token_loss * self.args.unique_token_weight + shared_token_loss * self.args.shared_token_weight
        else:
            total_loss = rec_loss + rec_loss_debias * self.args.debias_alpha

        return [total_loss, rec_loss, rec_loss_debias, unique_token_loss * self.args.unique_token_weight,
                shared_token_loss * self.args.shared_token_weight]
    def cal_debias_loss(self, scores_student, scores_teacher, target_item, negative_items):
        if self.args.distill_type == 1: # pair
            scores_student_target = torch.gather(scores_student, dim=-1, index=target_item.unsqueeze(-1))
            scores_teacher_target = torch.gather(scores_teacher, dim=-1, index=target_item.unsqueeze(-1))
            scores_debias_target_positive = scores_teacher_target > scores_teacher
            scores_debias_target_negative = scores_teacher_target < scores_teacher
            bpr_loss_positive = -F.logsigmoid(scores_student_target - scores_student) * scores_debias_target_positive
            bpr_loss_negative = -F.logsigmoid(scores_student - scores_student_target) * scores_debias_target_negative
            if scores_debias_target_positive.sum() > 0:
                bpr_loss_positive = bpr_loss_positive.sum() / scores_debias_target_positive.sum()
            else:
                bpr_loss_positive = 0
            if scores_debias_target_negative.sum() > 0:
                item2pop = torch.tensor(self.args.item2pop, device=scores_student.device)[: self.args.item_count]
                pop_weight = item2pop[negative_items.view(-1)].view(negative_items.size())
                pop_weight = 1 / (pop_weight + 5)
                pop_weight = pop_weight * scores_debias_target_negative
                pop_weight = pop_weight / pop_weight.sum() * scores_debias_target_negative.sum()
                bpr_loss_negative = bpr_loss_negative * pop_weight
                bpr_loss_negative = bpr_loss_negative.sum() / scores_debias_target_negative.sum()
            else:
                bpr_loss_negative = 0
            return (bpr_loss_positive + bpr_loss_negative) / 2
        elif self.args.distill_type == 2: # hard
            teacher_label = scores_teacher.max(dim=-1)[1]
            distill_loss = F.cross_entropy(scores_student, teacher_label)
            return distill_loss.half()
        elif self.args.distill_type == 3: # soft
            distribution_student = F.log_softmax(scores_student, dim=-1)
            distribution_teacher = F.softmax(scores_teacher, dim=-1)
            distill_loss = F.kl_div(distribution_student, distribution_teacher, reduction='batchmean')
            return distill_loss.half()

    def valid_step(self, inputs):
        seq_output = self.get_embedding(
            input_ids=inputs['sequence_input_ids'],
            attention_mask=inputs['sequence_attention_mask'],
            tfidf_scores=inputs['sequence_tfidf_scores'],
            pop_scores=inputs['sequence_pop_scores'],
            tail_scores=inputs['sequence_tail_scores']
        )
        seq_cls, _, _= seq_output
        item_cls = self.item_embs[inputs['negative_items']].to(seq_cls.device)
        with autocast(device_type='cuda', enabled=False):
            seq_cls = seq_cls.float().unsqueeze(-1)
            scores = seq_cls.squeeze(-1) @ self.item_embs.float().t()
            label = inputs['target_iid']

        return scores, label

    @torch.no_grad()
    def generate_embs(self, item_tokens):
        del self.item_embs
        torch.cuda.empty_cache()
        info(f"GPU:{self.args.gpu} Generating Emebedding")
        item_ids = item_tokens['item_ids']
        item_attn = item_tokens['item_attn']
        # 新增：获取真实的TF-IDF和流行度分数
        item_tfidf_scores = item_tokens['item_tfidf_scores']
        item_pop_scores = item_tokens['item_pop_scores']
        item_tail_scores = item_tokens['item_tail_scores']
        device = next(self.parameters()).device

        item_embs = []
        batch_size = 128
        for start_idx in range(0, item_ids.size()[0], batch_size):
            batch_item_ids = item_ids[start_idx: start_idx + batch_size].to(device)
            batch_item_attn = item_attn[start_idx: start_idx + batch_size].to(device)
            # 修复：使用真实的TF-IDF和流行度分数
            batch_tfidf_scores = item_tfidf_scores[start_idx: start_idx + batch_size].to(device)
            batch_pop_scores = item_pop_scores[start_idx: start_idx + batch_size].to(device)
            batch_tail_scores = item_tail_scores[start_idx: start_idx + batch_size].to(device)
            batch_output = self.get_embedding(
                input_ids=batch_item_ids,
                attention_mask=batch_item_attn,
                tfidf_scores=batch_tfidf_scores,
                pop_scores=batch_pop_scores,
                tail_scores=batch_tail_scores
            )
            batch_item_embs, _, _ = batch_output
            item_embs.append(batch_item_embs.cpu())
            torch.cuda.empty_cache()
        self.item_embs = torch.cat([x.to(device) for x in item_embs], dim=0)
        assert self.item_embs.size()[0] == item_ids.size()[0]

    def gather_indexes(self, output, gather_index):

        gather_index = gather_index.long()
        max_index = output.shape[1] - 1
        gather_index = torch.clamp(gather_index, 0, max_index)

        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

from torch.utils.data import Dataset, DataLoader
import pickle
import torch
from tqdm import tqdm
import os
import numpy as np
import random
import math
import copy
from html import unescape
from sklearn.feature_extraction.text import TfidfVectorizer

class DataSequential(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.length = 0
        self.data = None
        self.max_seq_length = args.max_seq_length
        self.max_item_tokens = 32
        self.max_token_length = args.max_token_length
        self.item_title_list = None
        self.candidate_index = []
        self.item_count = max(list(pickle.load(open(f"{args.data_path}/{args.dataset}/iid2asin.pkl", 'rb')).keys())) + 1
        self.args.item_count = self.item_count

        # 对每个用户的完整历史行为序列进行滑动窗口式的样本构建，然后根据时间顺序划分训练、验证和测试集
        self.load_data()
        self.item_title_tokens = None
        self.target2seqidx = None
        self.target2seqidx_copy = None
        self.target2pop = None
        self.item2pop = None
        self.token_tfidf = None  # Added: Store token TF-IDF scores
        self.token_pop_ratios = None  # 新增：token在流行物品中的出现比例
        self.token_tail_ratios = None  # 新增：token在长尾物品中的出现比例

        # 对每个物品的标题进行 tokenization
        self.tokenize_item_titles()
        self.sample_valid(self.data)
        self.candi_item_attention_mask = None
        self.candi_item_input_ids = None
        self.generate_cate_items()
        self.generate_target2seqidx()
        # 新增：记录每个 item 是否为流行物品
        self.item2is_popular = None
        self.generate_item2pop()

        # Added: Generate TF-IDF and token popularity scores
        self.generate_tfidf_scores()

        # 新增：计算token在流行和长尾物品中的出现比例
        self.generate_token_pop_tail_ratios()

        # 新增：计算全局均值和标准差
        self.normalize_scores()

    def normalize_scores(self):
        """归一化TF-IDF分数"""
        # 收集所有TF-IDF分数用于计算全局统计信息
        all_tfidf_scores = list(self.token_tfidf_dict.values())

        # 计算均值和标准差用于标准化
        self.tfidf_mean = np.mean(all_tfidf_scores)
        self.tfidf_std = np.std(all_tfidf_scores) + 1e-8  # 避免除零

        # ratio值不需要标准化，保持在[0,1]范围内

    # def generate_token_pop_tail_ratios(self):
    #     """计算每个token在流行物品和长尾物品中的出现比例"""
    #     from collections import defaultdict
    #
    #     # 统计每个token在流行物品和长尾物品中的出现次数
    #     token_in_popular_count = defaultdict(int)
    #     token_in_tail_count = defaultdict(int)
    #     token_total_count = defaultdict(int)
    #
    #     # 遍历所有物品，统计token出现情况
    #     for item_id, tokens in enumerate(self.item_title_tokens):
    #         is_popular = self.item2is_popular[item_id]
    #         unique_tokens = set(tokens)  # 每个物品中token只计算一次
    #
    #         for token in unique_tokens:
    #             token_total_count[token] += 1
    #             if is_popular:
    #                 token_in_popular_count[token] += 1
    #             else:
    #                 token_in_tail_count[token] += 1
    #
    #     # 计算比例
    #     token_pop_ratios = {}
    #     token_tail_ratios = {}
    #
    #     for token in token_total_count:
    #         total_count = token_total_count[token]
    #         pop_count = token_in_popular_count[token]
    #         tail_count = token_in_tail_count[token]
    #
    #         # 计算在流行物品中出现的比例
    #         token_pop_ratios[token] = pop_count / total_count
    #         # 计算在长尾物品中出现的比例
    #         token_tail_ratios[token] = tail_count / total_count
    #
    #     self.token_pop_ratios = token_pop_ratios
    #     self.token_tail_ratios = token_tail_ratios

    def generate_token_pop_tail_ratios(self):
        """计算每个token在流行物品和长尾物品中的出现比例"""
        from collections import defaultdict

        # 统计每个token在流行物品和长尾物品中的出现次数
        token_in_popular_count = defaultdict(int)
        token_in_tail_count = defaultdict(int)
        token_total_count = defaultdict(int)

        # 遍历所有物品，统计token出现情况
        for item_id, tokens in enumerate(self.item_title_tokens):
            is_popular = self.item2is_popular[item_id]
            unique_tokens = set(tokens)  # 每个物品中token只计算一次

            for token in unique_tokens:
                token_total_count[token] += 1
                if is_popular:
                    token_in_popular_count[token] += 1
                else:
                    token_in_tail_count[token] += 1

        # 计算标签：1=流行独有，2=长尾独有，3=共享
        token_pop_ratios = {}
        token_tail_ratios = {}

        for token in token_total_count:
            in_popular = token_in_popular_count[token] > 0
            in_tail = token_in_tail_count[token] > 0

            # 根据token出现的情况分配标签
            if in_popular and not in_tail:
                # 只在流行物品中出现
                token_pop_ratios[token] = 1
                token_tail_ratios[token] = 1
            elif in_tail and not in_popular:
                # 只在长尾物品中出现
                token_pop_ratios[token] = 2
                token_tail_ratios[token] = 2
            else:
                # 在流行和长尾物品中都有出现
                token_pop_ratios[token] = 3
                token_tail_ratios[token] = 3

        self.token_pop_ratios = token_pop_ratios
        self.token_tail_ratios = token_tail_ratios

        # 添加统计信息：计算流行物品中独有、共享token的占比
        self._analyze_popular_item_token_distribution_instances(token_in_popular_count, token_in_tail_count)

    def _analyze_popular_item_token_distribution_instances(self, token_in_popular_count, token_in_tail_count):
        """分析流行物品中独有token和共享token的分布（基于token实例数量）"""
        # 统计流行物品中的token实例
        popular_unique_token_instances = 0  # 只在流行物品中出现的token实例数
        shared_token_instances = 0  # 在流行和长尾物品中都出现的token实例数
        total_popular_token_instances = 0  # 流行物品中的所有token实例数

        # 遍历所有在流行物品中出现的token实例
        for token, popular_count in token_in_popular_count.items():
            total_popular_token_instances += popular_count
            if token_in_tail_count[token] > 0:
                # 在长尾物品中也出现，是共享token实例
                shared_token_instances += popular_count
            else:
                # 只在流行物品中出现，是流行独有token实例
                popular_unique_token_instances += popular_count

        # 计算占比
        if total_popular_token_instances > 0:
            popular_unique_ratio = popular_unique_token_instances / total_popular_token_instances
            shared_ratio = shared_token_instances / total_popular_token_instances

            print(f"\n=== 流行物品Token实例分布统计 ===")
            print(f"流行物品中独有token实例数: {popular_unique_token_instances}")
            print(f"流行物品中共享token实例数: {shared_token_instances}")
            print(f"流行物品中总token实例数: {total_popular_token_instances}")
            print(f"流行物品中独有token实例占比: {popular_unique_ratio:.4f} ({popular_unique_ratio * 100:.2f}%)")
            print(f"流行物品中共享token实例占比: {shared_ratio:.4f} ({shared_ratio * 100:.2f}%)")
            print("=" * 35)

            # 保存统计信息供后续使用
            self.popular_item_token_stats = {
                'popular_unique_instances': popular_unique_token_instances,
                'shared_instances': shared_token_instances,
                'total_instances': total_popular_token_instances,
                'popular_unique_ratio': popular_unique_ratio,
                'shared_ratio': shared_ratio
            }
        else:
            self.popular_item_token_stats = {
                'popular_unique_instances': 0,
                'shared_instances': 0,
                'total_instances': 0,
                'popular_unique_ratio': 0.0,
                'shared_ratio': 0.0
            }
    def get_all_training_example(self):
        train_examples = []
        for item in range(self.length):
            item_inputs = self.generate_example_input(self.data[item], item)
            train_examples.append([item_inputs[3], item_inputs[2]])
        return train_examples, self.get_items_tokens()

    def generate_item2pop(self):
        review_datas = pickle.load(open(f"{self.args.data_path}/{self.args.dataset}/review_datas.pkl", 'rb'))
        item2popularity = [0] * self.item_count
        for user in tqdm(review_datas.keys(), desc='Splitting Train/Valid/Test'):
            for i in range(1, len(review_datas[user])):
                review = review_datas[user][i]
                if i < len(review_datas[user]) - 2:
                    item2popularity[review[0]] += 1
        self.item2pop = item2popularity
        self.args.item2pop = self.item2pop

        # 新增：基于流行度排序，标记流行/长尾物品
        pop_threshold = np.percentile(item2popularity, 80)  # 前 20% 为流行物品
        self.item2is_popular = [1 if pop >= pop_threshold else 0 for pop in item2popularity]
        self.args.item2is_popular = self.item2is_popular

    def generate_target2seqidx(self):
        if self.mode != 'train':
            return
        target2seqidx = [[] for _ in range(self.item_count)]
        for idx in range(len(self.data)):
            target_iid = self.data[idx][1]
            target2seqidx[target_iid].append(idx)
        self.target2seqidx = target2seqidx
        target2pop = [math.pow(len(x), self.args.sample_alpha) for x in target2seqidx]
        target2pop_sum = sum(target2pop)
        self.target2pop = [x / target2pop_sum for x in target2pop]
        self.target2seqidx_copy = copy.deepcopy(target2seqidx)

    def get_item_token(self, idx, sample=False):
        item_token = self.item_title_tokens[idx]
        if not sample or self.args.token_ratio == 1.0 or self.mode != 'train':
            return item_token
        sample_token = (torch.rand([len(item_token)]) < self.args.token_ratio).nonzero().squeeze(1)
        item_token = [item_token[t_idx] for t_idx in sample_token]
        return item_token

    def sample_valid(self, datas):
        if self.args.valid_ratio == 1 or self.mode != 'valid':
            return
        import random
        random.seed(42)
        sample_idx = random.sample(list(range(len(datas))), int(len(datas) * self.args.valid_ratio))
        sample_idx.sort()
        new_datas = []
        for idx in sample_idx:
            new_datas.append(datas[idx])
        self.length = len(new_datas)
        self.data = new_datas

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        target_item = 0
        if self.mode == 'train' and self.args.sample_alpha != 1.0:
            while target_item == 0:
                if len(self.candidate_index) == 0:
                    self.candidate_index = np.random.choice(self.item_count, size=10000, p=self.target2pop).tolist()
                target_item = self.candidate_index.pop()
                if len(self.target2seqidx[target_item]) == 0:
                    target_item = 0
                else:
                    if len(self.target2seqidx_copy[target_item]) == 0:
                        self.target2seqidx_copy[target_item] = self.target2seqidx[target_item] + []
                        random.shuffle(self.target2seqidx_copy[target_item])
                    item = self.target2seqidx_copy[target_item].pop()

        example_input = self.generate_example_input(self.data[item], item)
        example_input.append(item)
        return example_input

    def load_data(self):
        review_datas = pickle.load(open(f"{self.args.data_path}/{self.args.dataset}/review_datas.pkl", 'rb'))
        train_data = []
        valid_data = []
        test_data = []

        for user in tqdm(review_datas.keys(), desc='Splitting Train/Valid/Test'):
            seq_iid_list = [review_datas[user][0][0]]
            seq_iid_cate_list = [review_datas[user][0][2]]
            for i in range(1, len(review_datas[user])):
                target_iid = review_datas[user][i][0]
                target_iid_cate = review_datas[user][i][2]
                if i < len(review_datas[user]) - 2:
                    train_data.append([seq_iid_list, target_iid, seq_iid_cate_list, target_iid_cate])
                elif i == len(review_datas[user]) - 2:
                    valid_data.append([seq_iid_list, target_iid, seq_iid_cate_list, target_iid_cate])
                elif i == len(review_datas[user]) - 1:
                    test_data.append([seq_iid_list, target_iid, seq_iid_cate_list, target_iid_cate])
                else:
                    raise NotImplementedError
                seq_iid_list = seq_iid_list + [review_datas[user][i][0]]
                seq_iid_cate_list = seq_iid_cate_list + [review_datas[user][i][2]]

                seq_iid_list = seq_iid_list[-self.max_seq_length:]
                seq_iid_cate_list = seq_iid_cate_list[-self.max_seq_length:]

        if self.mode == 'train':
            self.data = train_data
        elif self.mode == 'valid':
            self.data = valid_data
        elif self.mode == 'test':
            self.data = test_data
        else:
            raise NotImplementedError
        self.length = len(self.data)

    def generate_cate_items(self):
        candi_item_input_ids = []
        candi_item_attention_mask = []
        fp_tokens = self.max_item_tokens + 1
        for idx in range(self.item_count):
            candi_tokens = self.get_item_token(idx, True) + [self.tokenizer.eos_token_id]
            pad_len = fp_tokens - len(candi_tokens)
            candi_item_input_ids.append(candi_tokens + [0] * pad_len)
            candi_item_attention_mask.append((len(candi_tokens) * [1] + [0] * pad_len))
        self.candi_item_input_ids = candi_item_input_ids
        self.candi_item_attention_mask = candi_item_attention_mask

    def tokenize_item_titles(self):
        item_metas = pickle.load(open(f"{self.args.data_path}/{self.args.dataset}/meta_datas.pkl", 'rb'))
        iid2asin = pickle.load(open(f"{self.args.data_path}/{self.args.dataset}/iid2asin.pkl", 'rb'))
        id_prefix = 'id:'
        title_prefix = 'title:'
        item_title_list = ['None'] * self.item_count
        for iid, asin in iid2asin.items():
            item_title = item_metas[asin]['title'] if ('title' in item_metas[asin].keys() and item_metas[asin]['title']) else 'None'
            item_title = item_title.replace('&', '')
            item_title = id_prefix + ' ' + str(iid) + ' ' + title_prefix + ' ' + item_title + ', '
            item_title_list[iid] = item_title

        item_max_tokens = self.max_item_tokens
        item_title_tokens = []
        for start in tqdm(range(0, len(item_title_list), 32), desc='Tokenizing'):
            tokenized_text = self.tokenizer(item_title_list[start: start + 32],
                                            truncation=True,
                                            max_length=item_max_tokens,
                                            padding=False,
                                            add_special_tokens=False,
                                            return_tensors=None)
            item_title_tokens.extend(tokenized_text['input_ids'])
        self.item_title_tokens = item_title_tokens
        template1 = "Here is the visit history list of user: "
        template2 = " recommend next item "
        self.template1_ids = self.tokenizer.encode(template1, add_special_tokens=False, truncation=False)
        self.template2_ids = self.tokenizer.encode(template2, add_special_tokens=False, truncation=False)
        self.item_title_list=item_title_list
    def generate_tfidf_scores(self):
        """Generate global TF-IDF scores for each unique token in the dataset manually."""
        from collections import defaultdict
        import math

        # Step 1: 统计每个 token 的 TF（Term Frequency）
        token_tf = defaultdict(float)  # token_id -> total count
        token_df = defaultdict(int)  # token_id -> doc count
        total_tokens = 0

        # Step 2: 统计每个 token 的 DF（Document Frequency）
        for tokens in self.item_title_tokens:
            doc_token_count = defaultdict(int)
            for token in tokens:
                doc_token_count[token] += 1
                total_tokens += 1

            # 每个文档中只计一次 DF
            unique_tokens_in_doc = set(tokens)
            for token in unique_tokens_in_doc:
                token_df[token] += 1

            # 累加 TF（Term Frequency）
            for token, count in doc_token_count.items():
                token_tf[token] += count

        # Step 3: 计算全局 IDF（Inverse Document Frequency）
        N = len(self.item_title_tokens)  # 总文档数（item 数量）
        token_idf = {}
        for token in token_df:
            token_idf[token] = math.log(N / (1 + token_df[token]))

        # Step 4: 计算 TF-IDF = TF × IDF
        self.token_tfidf_dict = {}
        for token in token_tf:
            self.token_tfidf_dict[token] = token_tf[token] * token_idf[token]

    def generate_example_input(self, example, example_idx):
        seq_iid_list, target_iid = example[0], example[1]
        sequence_input_ids = []

        # 记录每个物品在序列中的token位置范围
        seq_item_positions = []

        for seq_iid in seq_iid_list:
            seq_i_tokens = self.get_item_token(seq_iid, True)
            start_pos = len(sequence_input_ids)
            sequence_input_ids.extend(seq_i_tokens)
            end_pos = len(sequence_input_ids)
            seq_item_positions.append((start_pos, end_pos))

        sequence_input_ids = self.template1_ids + sequence_input_ids + self.template2_ids
        # 更新物品位置以考虑模板token的偏移
        offset = len(self.template1_ids)
        seq_item_positions = [(start + offset, end + offset) for start, end in seq_item_positions]

        sequence_attention_mask = [1] * len(sequence_input_ids)

        sequence_input_ids = sequence_input_ids + [self.tokenizer.eos_token_id]
        sequence_attention_mask.append(1)

        if self.mode == 'train':
            negative_items = random.sample(range(1, self.item_count), self.args.train_nega_count)
            target_position = 0
        else:
            negative_items = random.sample(range(1, self.item_count), self.args.nega_count)
            target_position = 0

        negative_items = negative_items[0:target_position] + [target_iid] + negative_items[target_position:]
        negative_items_pop = [self.item2pop[x] for x in negative_items]

        # 添加用户历史物品的流行度标签
        seq_items_pop = [self.item2pop[x] for x in seq_iid_list]
        # 根据已有的item2is_popular判断用户历史物品是否为流行物品
        seq_is_popular = [self.item2is_popular[x] for x in seq_iid_list]

        # 保证无论在任何模式下都生成真实的token序列，确保输入信息一致性
        candi_item_input_ids = [self.candi_item_input_ids[x] for x in negative_items]
        candi_item_attention_mask = [self.candi_item_attention_mask[x] for x in negative_items]

        return [candi_item_input_ids, candi_item_attention_mask, sequence_attention_mask, sequence_input_ids,
                target_position, target_iid, negative_items, negative_items_pop, seq_is_popular, seq_item_positions]

    def get_items_tokens(self):
        item_ids = []
        item_attn = []
        item_tfidf_scores = []  # TF-IDF分数
        item_pop_scores = []  # 新增：token在流行物品中的出现比例
        item_tail_scores = []  # 新增：token在长尾物品中的出现比例
        fp_tokens = self.max_item_tokens + 1
        for iid in range(len(self.item_title_tokens)):
            item_tokens = self.get_item_token(iid) + [self.tokenizer.eos_token_id]
            pad_len = fp_tokens - len(item_tokens)
            item_ids.append(item_tokens + [0] * pad_len)
            item_attn.append(len(item_tokens) * [1] + pad_len * [0])

            # 计算物品的TF-IDF分数和两个新的比例特征
            tfidf = [self.token_tfidf_dict.get(token, 0.0) for token in item_tokens] + [0.0] * pad_len
            pop_ratio = [self.token_pop_ratios.get(token, 0.0) for token in item_tokens] + [0.0] * pad_len
            tail_ratio = [self.token_tail_ratios.get(token, 0.0) for token in item_tokens] + [0.0] * pad_len

            # 只对TF-IDF进行标准化处理
            tfidf = [(x - self.tfidf_mean) / self.tfidf_std for x in tfidf]
            item_tfidf_scores.append(tfidf)
            item_pop_scores.append(pop_ratio)
            item_tail_scores.append(tail_ratio)

        return {'item_ids': torch.LongTensor(item_ids),
                'item_attn': torch.LongTensor(item_attn),
                'item_tfidf_scores': torch.FloatTensor(item_tfidf_scores),
                'item_pop_scores': torch.FloatTensor(item_pop_scores),
                'item_tail_scores': torch.FloatTensor(item_tail_scores)}

    def collate_fn(self, batch_data):
        item_input_ids = []
        item_attention_mask = []
        sequence_attention_mask = []
        sequence_input_ids = []
        target_position = []
        target_iid = []
        example_index = []
        negative_items = []
        negative_items_pop = []

        item_tfidf_scores = []
        item_pop_scores = []
        item_tail_scores = []  # 新增
        sequence_tfidf_scores = []
        sequence_pop_scores = []
        sequence_tail_scores = []  # 新增
        item_is_popular = []  # 新增：记录 item 是否为流行物品
        seq_is_popular = []  # 新增：记录序列中物品是否为流行物品
        seq_item_positions = []  # 新增：记录序列中每个物品的token位置

        # max_seq_length计算一个batch中所有样本的序列长度的最大值
        max_seq_length = max(len(x[2]) for x in batch_data)
        max_item_length = self.max_item_tokens + 1

        for example in batch_data:
            item_input_ids.extend(example[0])
            item_attention_mask.extend(example[1])

            seq_pad_len = max_seq_length - len(example[2])
            sequence_attention_mask.append(example[2] + seq_pad_len * [0])
            sequence_input_ids.append(example[3] + seq_pad_len * [0])
            target_position.append(example[4])
            target_iid.append(example[5])
            negative_items.append(example[6])
            negative_items_pop.append(example[7])

            example_index.append(example[-1])

            # 添加序列中物品的位置信息（保持原始长度，不进行填充）
            if len(example) > 9:  # 如果有seq_item_positions信息
                positions = example[9]  # seq_item_positions在example[9]中
            else:
                positions = []  # 默认空列表
            seq_item_positions.append(positions)

            # Item token 的 TF-IDF 和两个新的比例特征
            # 无论在训练还是验证阶段都处理，确保一致性
            item_tfidf = []
            item_pop = []  # pop_ratio
            item_tail = []  # tail_ratio
            item_popular_labels = []
            for idx, item_id in enumerate(example[6]):  # 使用索引而不是查找
                tokens = example[0][idx][:max_item_length]  # 直接使用索引获取tokens
                attention_mask = example[1][idx][:max_item_length]  # 获取对应的attention mask

                # 根据attention mask判断哪些位置是padding
                tfidf_raw = []
                pop_raw = []  # pop_ratio
                tail_raw = []  # tail_ratio
                for i, token in enumerate(tokens):
                    if i < len(attention_mask) and attention_mask[i] == 1:
                        # 有效位置使用真实值
                        tfidf_raw.append(self.token_tfidf_dict.get(token, 0.0))
                        pop_raw.append(self.token_pop_ratios.get(token, 0.0))
                        tail_raw.append(self.token_tail_ratios.get(token, 0.0))
                    else:
                        # padding位置使用-10作为占位符
                        tfidf_raw.append(0.0)
                        pop_raw.append(0.0)
                        tail_raw.append(0.0)

                # 只对TF-IDF进行标准化处理
                tfidf = [(x - self.tfidf_mean) / self.tfidf_std for x in tfidf_raw]
                item_tfidf.append(tfidf)
                item_pop.append(pop_raw)
                item_tail.append(tail_raw)
                item_popular_labels.append(self.item2is_popular[item_id])
            item_tfidf_scores.extend(item_tfidf)
            item_pop_scores.extend(item_pop)
            item_tail_scores.extend(item_tail)
            item_is_popular.extend(item_popular_labels)

            # Sequence token 的 TF-IDF 和两个新的比例特征
            seq_tokens = example[3]
            seq_tfidf_raw = []
            seq_pop_raw = []  # pop_ratio
            seq_tail_raw = []  # tail_ratio
            for token in seq_tokens[:max_seq_length]:
                tfidf_score = self.token_tfidf_dict.get(token, 0.0)
                pop_score = self.token_pop_ratios.get(token, 0.0)
                tail_score = self.token_tail_ratios.get(token, 0.0)
                seq_tfidf_raw.append(tfidf_score)
                seq_pop_raw.append(pop_score)
                seq_tail_raw.append(tail_score)

            seq_tfidf_raw += [0.0] * (max_seq_length - len(seq_tfidf_raw))
            seq_pop_raw += [.0] * (max_seq_length - len(seq_pop_raw))
            seq_tail_raw += [0.0] * (max_seq_length - len(seq_tail_raw))

            # 只对TF-IDF进行标准化处理
            seq_tfidf = [(x - self.tfidf_mean) / self.tfidf_std for x in seq_tfidf_raw]
            seq_pop = seq_pop_raw  # ratio值保持原始
            seq_tail = seq_tail_raw  # ratio值保持原始

            sequence_tfidf_scores.append(seq_tfidf)
            sequence_pop_scores.append(seq_pop)
            sequence_tail_scores.append(seq_tail)

            # 添加序列中物品的流行度信息
            seq_is_popular.append(example[8])  # seq_is_popular信息在example[8]中
        return {
            'item_input_ids': torch.LongTensor(item_input_ids),
            'item_attention_mask': torch.LongTensor(item_attention_mask),
            'sequence_attention_mask': torch.LongTensor(sequence_attention_mask),
            'sequence_input_ids': torch.LongTensor(sequence_input_ids),
            'target_position': torch.LongTensor(target_position),
            'target_iid': torch.LongTensor(target_iid),
            'example_index': torch.LongTensor(example_index),
            'negative_items': torch.LongTensor(negative_items),
            'negative_items_pop': torch.FloatTensor(negative_items_pop),

            'item_tfidf_scores': torch.FloatTensor(item_tfidf_scores),
            'item_pop_scores': torch.FloatTensor(item_pop_scores),
            'item_tail_scores': torch.FloatTensor(item_tail_scores),  # 新增
            'sequence_tfidf_scores': torch.FloatTensor(sequence_tfidf_scores),
            'sequence_pop_scores': torch.FloatTensor(sequence_pop_scores),
            'sequence_tail_scores': torch.FloatTensor(sequence_tail_scores),  # 新增
            'item_is_popular': torch.LongTensor(item_is_popular),  # 新增
            'seq_is_popular': seq_is_popular,  # 新增
            'seq_item_positions': seq_item_positions  # 新增，保持原始长度
        }

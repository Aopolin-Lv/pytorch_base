# coding=utf-8
"""
@Time:          2020/11/2 7:01 下午
@Author:        Aopolin
@File:          test.py
@Contact:       aopolin.ii@gmail.com
@Description:
"""
import os
from pathlib import Path
import re
from collections import Counter
import numpy as np
import random
import torch
import torch.nn as nn
import json


class Config(object):
    """
    配置参数
    """
    data_name = "CR"
    status = 'train'  # 执行 train_eval or test, 默认执行train_eval
    output_folder = 'output_data/'  # 已处理的数据所在文件夹
    data_path = 'data/'  # 数据集所在路径
    emb_file = 'data/glove.6B.300d.txt'  # 预训练词向量所在路径
    pretrain_emb_file_suffix = "_pretrain_embed.pth"
    vocab_file_suffix = "_vocab.json"
    emb_format = 'glove'  # embedding format: word2vec/glove
    min_word_freq = 1  # 最小词频
    max_len = 40  # 采样最大长度
    MAX_VOCAB_SIZE = 30000

    # 训练参数
    epochs = 120  # epoch数目，除非early stopping, 先开20个epoch不微调,再开多点epoch微调
    batch_size = 64  # batch_size
    lr = 1e-4  # 如果要微调时，学习率要小于1e-3,因为已经是很优化的了，不用这么大的学习率
    weight_decay = 1e-5  # 权重衰减率
    decay_epoch = 15  # 多少个epoch后执行学习率衰减
    improvement_epoch = 30  # 多少个epoch后执行early stopping
    print_freq = 100  # 每隔print_freq个iteration打印状态
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型参数
    model_name = 'TextAttnBiLSTM'  # 模型名
    class_num = 2  # 分类类别
    embed_dropout = 0.3  # dropout
    model_dropout = 0.5  # dropout
    fc_dropout = 0.5  # dropout
    num_layers = 2  # LSTM层数
    embed_dim = 128  # 未使用预训练词向量的默认值
    use_embed = True  # 是否使用预训练
    use_gru = True  # 是否使用GRU
    grad_clip = 4.  # 梯度裁剪阈值


# 全局配置参数
opt = Config()


class Rule:
    # 正则表达式过滤特殊符号用空格符占位，双引号、单引号、句点、逗号
    pat_letter = re.compile(r'[^a-zA-Z \']+')  # 保留'
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)  # 还原常见缩写单词
    pat_s = re.compile("([a-zA-Z])(\'s)")  # 处理类似于这样的缩写today’s
    pat_not = re.compile("([a-zA-Z])(n\'t)")  # not的缩写
    pat_would = re.compile("([a-zA-Z])(\'d)")  # would的缩写
    pat_will = re.compile("([a-zA-Z])(\'ll)")  # will的缩写
    pat_am = re.compile("([I|i])(\'m)")  # am的缩写
    pat_are = re.compile("([a-zA-Z])(\'re)")  # are的缩写
    pat_ve = re.compile("([a-zA-Z])(\'ve)")  # have的缩写


def Replace_Abbreviations(sentence):
    """
    替换缩写字符并且进行大写->小写
    :param sentence:  待清洗语料
    :return: cleaned_sentence: 清洗完成后的语料
    """
    cleaned_sentence = sentence
    cleaned_sentence = Rule.pat_letter.sub(' ', cleaned_sentence).strip().lower()
    cleaned_sentence = Rule.pat_is.sub(r"\1 is", cleaned_sentence)  # 其中\1是匹配到的第一个group
    cleaned_sentence = Rule.pat_s.sub(r"\1 ", cleaned_sentence)
    cleaned_sentence = Rule.pat_not.sub(r"\1 not", cleaned_sentence)
    cleaned_sentence = Rule.pat_would.sub(r"\1 would", cleaned_sentence)
    cleaned_sentence = Rule.pat_will.sub(r"\1 will", cleaned_sentence)
    cleaned_sentence = Rule.pat_am.sub(r"\1 am", cleaned_sentence)
    cleaned_sentence = Rule.pat_are.sub(r"\1 are", cleaned_sentence)
    cleaned_sentence = Rule.pat_ve.sub(r"\1 have", cleaned_sentence)
    cleaned_sentence = cleaned_sentence.replace('\'', ' ')
    return cleaned_sentence


def Load_Data(input_file_name: str):
    """
    从文件读入数据
    :param input_file_name: 数据文件名称
    :return: sentence: 读取的数据
    """
    sentence = []
    read_type_list = ["train", "dev", "test"]
    path = Path("data/" + input_file_name + "/" + input_file_name.lower() + ".train.txt")

    # 读入数据
    with open(path, "r", encoding="utf-8") as f:
        for l in f:
            line = l.strip().split("|||")
            if len(line) < 2:
                print("error")
            sent = Replace_Abbreviations(line[1])  # 清洗数据
            sent = Word_Tokenize(sent)  # 分词
            label = line[0]
            sentence.append((sent, label))

    return sentence


def Word_Tokenize(text):
    """
    将句子分词
    :param text:
    :return:
    """
    return text.split()


def Create_Wordlist(text: list, max_vocab_size: int):
    """
    创建词表，统计词频，词数，标签数
    :param text: 句子列表
    :param max_vocab_size:
    :return: word_frequency: 词表, label_count: 标签数
    """
    vocab_dict = dict(Counter(text).most_common(max_vocab_size - 1))
    vocab_dict["<unk>"] = len(text) - np.sum(list(vocab.values()))
    return vocab_dict


def Build_Dict(text, max_vocab_size=50000):
    """
    创建词表
    :param text: 文本
    :param max_vocab_size:
    :return:
    """
    word_count = Counter()
    for word in text:
        word_count[word] += 1
    ls = word_count.most_common(max_vocab_size)
    total_words = len(ls) + 2
    vocab_dict = dict()
    vocab_dict["<unk>"] = 0
    vocab_dict["<pad>"] = 1
    for index, w in enumerate(ls):
        vocab_dict[w[0]] = index + 2

    Save_Vocab(vocab_dict, opt.output_folder, opt.data_name)
    print("Save vocab success")
    return vocab_dict, total_words


def Idx2Word(vocab_dict: dict):
    """
    建立 index->word 索引表
    :param vocab_dict:
    :return:
    """
    return [word for word in vocab_dict.keys()]


def Word2Idx(vocab_dict: dict):
    """
    建立 word->index 索引表
    :param vocab_dict:
    :return:
    """
    idx_to_word = Idx2Word(vocab_dict)
    return {word: i for i, word in enumerate(idx_to_word)}


def Word_Counts_and_Frequency(vocab: dict):
    """
    返回词频
    :param vocab:
    :return: word_counts: 单词出现次数
    :return: word_freqs: 单词出现频率
    """
    # 将vocab词表中的单词出现的次数排成一个np数组
    word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
    # 对word_counts数组进行词频计算
    word_freqs = word_counts / np.sum(word_counts)

    return word_counts, word_freqs


def ConvertWord2Idx(sentences_label_list, vocab):
    """
    根据词表，将句子列表中的word全部转成数字
    :param sentences_label_list:
    :param vocab:
    :return:
    """
    output = []
    for idx, (sent, label) in enumerate(sentences_label_list):
        out_sentences = [vocab.get(word, 0) for word in sent]
        output.append([out_sentences, label])
    return output


def Build_Batch(data_list, batch_size, sorted_by_len=True, shuffle=True):
    def Get_Minibatches(data_len, minibatch_size, shuffle=True):
        # 得到按照batch大小划分的index_list,如batch_size == 2，idx_list = [0, 2, 4, 6...]
        idx_list = np.arange(0, data_len, batch_size)
        # 将idx_list随机化
        if shuffle:
            np.random.shuffle(idx_list)

        # 将train_list -> numpy矩阵
        minibatches = []
        for idx in idx_list:
            minibatches.append(np.arange(idx, min(idx + minibatch_size, data_len)))
        return minibatches

    def Fill_Data(sentences):
        lengths = [len(sen[0]) for sen in sentences]  # 一个batch里每一个sentence的长度组成一个list
        labels = [sen[1] for sen in sentences]
        n_samples = len(sentences)  # 其实是一个batch里有多少sample(有可能不足batch_size)
        max_len = np.max(lengths)

        # 用0矩阵初始化x(numpy matrix)
        x = np.zeros((n_samples, max_len)).astype("int32")
        # 将一个batch里每一个sentence的标签st -> numpy matrix
        x_labels = np.array(labels).astype("int32")

        for idx, sen in enumerate(sentences):
            x[idx, :lengths[idx]] = sen[0]
        return x, x_labels

    def Generate_Batch(data_list, batch_size, shuffle):
        minibatches = Get_Minibatches(len(data_list), batch_size, shuffle)
        result = []
        for minibatch in minibatches:
            mb_sent = [data_list[t] for t in minibatch]
            mb_x, mb_x_labels = Fill_Data(mb_sent)
            result.append((mb_x, mb_x_labels))
        return result

    if sorted_by_len:
        data_list = sorted(data_list, key=lambda k: len(k[0][0]))

    data_batch = Generate_Batch(data_list, batch_size, shuffle)
    return data_batch


def Init_Embeddings(embeddings):
    """
    使用均匀分布U(-bias, bias)来随机初始化
    :param embeddings: 词向量矩阵
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    nn.init.uniform_(embeddings, -bias, bias)


def Load_Embeddings(emb_file, vocab):
    """
    家在预训练文件，如glove.6B.300d.txt，每行是str
    :param emb_file: 预训练文件
    :param vocab: 词表
    :return:
    """
    cnt = 0  # 记录读入的词数
    with open(emb_file, "r", encoding="utf-8") as f:
        emb_dim = len(f.readline().split(" ")) - 1

    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    # 初始化词向量(对OOV进行随机初始化，即对那些在词表上的词但不在与训练词向量中的词)
    Init_Embeddings(embeddings)

    # 读入词向量文件
    with open(emb_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split(" ")
            emb_word = line[0]

            # 筛除空值并且转化成float类型
            embedding = list(map(lambda t: float(t),
                                 filter(lambda n: n and not n.isspace(), line[1:])))

            # 如果预训练的单词emb_word不在vocab表里，则直接进入下一个循环
            if emb_word not in vocab:
                continue
            else:
                cnt += 1

            embeddings[vocab[emb_word]] = torch.FloatTensor(embedding)

    Save_Pth(embeddings, emb_dim, opt.output_folder, opt.data_name)

    print("Number of words read: ", cnt)
    print("Number of OOV: ", len(vocab) - cnt)
    print("Save pretrain_embed success")

    return embeddings, emb_dim


def Save_Vocab(vocab, output_folder, data_name):
    """
    保存由训练集构建的词表
    :param vocab:
    :param output_folder:
    :param data_name:
    :return:
    """
    with open(os.path.join(output_folder, data_name + opt.vocab_file_suffix), "w") as j:
        json.dump(vocab, j)


def Load_Vocab(vocab_filename):
    with open(vocab_filename, "r") as j:
        vocab = json.load(j)
    return vocab


def Save_Pth(pretrain_embed, embed_dim, output_folder, data_name):
    """
    存储预训练的向量
    :param pretrain_embed: 预训练向量
    :param embed_dim: 预训练向量维度
    :param output_folder: 输出目录
    :param data_name: 文件名称
    :return : None
    """
    embed = dict()
    embed["pretrain_embed"] = pretrain_embed
    embed["embed_dim"] = embed_dim
    torch.save(embed, output_folder + data_name + opt.pretrain_emb_file_suffix)


def Load_Pth(embed_file_name):
    embed_file = torch.load(embed_file_name)
    return embed_file["pretrain_embed"], embed_file["embed_dim"]


# 读取数据
t = Load_Data(opt.data_name)

# 获得长句
text = ""
for i, (word_list, label) in enumerate(t):
    sentences = " ".join(word_list)
    text += sentences

# 创建词表
vocab_filename = os.path.join(opt.output_folder, opt.data_name + opt.vocab_file_suffix)
if not os.path.exists(vocab_filename):
    vocab, _ = Build_Dict(Word_Tokenize(text), opt.MAX_VOCAB_SIZE)
else:
    vocab = Load_Vocab(vocab_filename)

idx_to_word = Idx2Word(vocab)
word_to_idx = Word2Idx(vocab)
word_counts, word_freqs = Word_Counts_and_Frequency(vocab)

pretrain_emb_filename = os.path.join(opt.output_folder, opt.data_name + opt.pretrain_emb_file_suffix)
if not os.path.exists(pretrain_emb_filename):
    pretrain_emb, emb_dim = Load_Embeddings(opt.emb_file, vocab)
else:
    pretrain_emb, emb_dim = Load_Pth(pretrain_emb_filename)

sentences_list = ConvertWord2Idx(t, vocab)
data_batch = Build_Batch(sentences_list, opt.batch_size, sorted_by_len=False)

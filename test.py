# coding=utf-8

"""
@Time:          2020/11/2 7:01 下午
@Author:        Aopolin
@File:          test.py
@Contact:       aopolin.ii@gmail.com
@Description:
"""
from pathlib import Path
import re
from collections import Counter
import numpy as np
import random


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
    vocab = dict(Counter(text).most_common(max_vocab_size - 1))
    vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
    return vocab


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
    vocab = dict()
    vocab["<unk>"] = 0
    vocab["<pad>"] = 1
    for index, w in enumerate(ls):
         vocab[w[0]] = index + 2
    return vocab, total_words


def Idx2Word(vocab: dict):
    """
    建立 index->word 索引表
    :param vocab:
    :return:
    """
    return [word for word in vocab.keys()]


def Word2Idx(vocab: dict):
    """
    建立 word->index 索引表
    :param vocab:
    :return:
    """
    idx_to_word = Idx2Word(vocab)
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


def Build_Batch(data_list, batch_size, shuffle=True):
    def Get_Minibatches(data_len, minibatch_size, shuffle=True):
        # 得到按照batch大小划分的index_list,如batch_size == 2，idx_list = [0, 2, 4, 6...]
        idx_list = np.arange(0, data_len, batch_size)
        # 将idx_list随机化
        if shuffle:
            np.random.shuffle(idx_list)

        # 将train_list -> numpy矩阵
        minibatches = []
        for idx in idx_list:
            minibatches.append(np.arange(idx, min(idx+minibatch_size, data_len)))
        return minibatches

    def Fill_Data(sentences):
        lengths = [len(sen) for sen in sentences]   # 一个batch里每一个sentence的长度组成一个list
        n_samples = len(sentences)                  # 其实是一个batch里有多少sample(有可能不足batch_size)
        max_len = np.max(lengths)

        # 用0矩阵初始化x(numpy matrix)
        x = np.zeros((n_samples, max_len)).astype("int32")
        # 将一个batch里每一个sentence的长度组成的list -> numpy matrix
        x_lengths = np.array(lengths).astype("int32")

        for idx, sen in enumerate(sentences):
            x[idx, :lengths[idx]] = sen
        return x, x_lengths

    def Generate_Batch(data_list, batch_size, shuffle):
        minibatches = Get_Minibatches(len(data_list), batch_size, shuffle)
        result = []
        for minibatch in minibatches:
            mb_sent = [data_list[t] for t in minibatch]
            mb_x, mb_x_len = Fill_Data(mb_sent)
            result.append((mb_x, mb_x_len))
        return result

    data_batch = Generate_Batch(data_list, batch_size, shuffle)
    random.shuffle(data_batch)
    return data_batch


"""
相关参数定义
"""
MAX_VOCAB_SIZE = 3000

# 读取数据
t = Load_Data("CR")

# 获得长句
text = ""
for i, (word_list, label) in enumerate(t):
    sentences = " ".join(word_list)
    text += sentences

# 创建词表
vocab, _ = Build_Dict(Word_Tokenize(text), MAX_VOCAB_SIZE)
idx_to_word = Idx2Word(vocab)
word_to_idx = Word2Idx(vocab)
word_counts, word_freqs = Word_Counts_and_Frequency(vocab)

# 更新VOCAB_SIZE词表长度，以防不足
VOCAB_SIZE = len(idx_to_word)
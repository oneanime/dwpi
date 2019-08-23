import linecache
import pandas as pd
import os
from string import punctuation
import nltk
import re
from nltk.corpus import stopwords
import itertools
from collections import Counter
import numpy as np


class DataLoad(object):
    def __init__(self, path):
        # 文件夹绝对路径
        self.path = path
        # 获取文件名
        self.file_list = self.get_file(self.path)
        self.records_count = 0

    # 获取文件名
    def get_file(self, path):
        return [filename for filename in os.listdir(path)]

    # 提取整块记录
    def pt_to_er(self, line_list):
        pt_list = list()
        er_list = list()

        for i in range(len(line_list)):
            if re.match("^PT", line_list[i]):
                pt_list.append(i)
            if re.match("^ER", line_list[i]):
                er_list.append(i)

        df = pd.DataFrame(data=[pt_list, er_list]).T

        result = list()
        for row in df.iterrows():
            result.append("".join(line_list[row[1][0]:row[1][1] + int(1)]))
        return result

    # 获取所有记录
    def getAllRecords(self):
        all_records = list()
        for file in self.file_list:
            if file.split('.')[1] == "txt":
                full_path = os.path.join(self.path, file)
                line_list = linecache.getlines(full_path)
                # 从所有行中找PT-ER记录块，并返回为每个文件中的专利记录
                result_list = self.pt_to_er(line_list)
                all_records.extend(result_list)
                # self.save_db(result_list)
        self.records_count = len(all_records)
        return all_records


class ExtractFields(object):

    # 获取指定字段
    @staticmethod
    def extract_field(data, datafield):
        lines = data.split('\n')
        flag = False
        index = int()
        index1 = int()
        for i in range(len(lines)):
            if lines[i][0:2] == datafield:
                index = i
                flag = True
                continue
            if flag and (not lines[i][0].isspace()):
                index1 = i
                flag = False
        lines[index] = lines[index][2:]
        return lines[index:index1]

    # 获取指定所有字段
    @classmethod
    def extract_field_list(cls, data, datafield):
        return [cls.extract_field(d, datafield) for d in data]

    # 获取多个指定字段的值
    @classmethod
    def extract_fields(cls, data, *datafields):
        return {field: cls.extract_field(data, field) for field in datafields}

    # 获取所有的多个指定字段的值
    @classmethod
    def extract_fields_list(cls, data, *datafields):
        return [cls.extract_fields(d, datafields) for d in data]


class CleanFields(object):

    # 对字段AB中不符合的文本过滤
    @staticmethod
    def clean_ab(data):
        result = [i for i in data if not re.search(r':', i)]
        for c in result:
            if re.search(r'-', c):
                index = result.index(c)
                result[index] = re.split(r'-', c, maxsplit=1)[1]
        return result


class ProcessDwpiText(object):

    def leaves(self, toks, grammar, pos):
        chunker = nltk.RegexpParser(grammar)
        postoks = nltk.tag.pos_tag(toks)
        tree = chunker.parse(postoks)
        # tree.draw()
        for subtree in tree.subtrees(lambda t: t.label() == pos):
            yield subtree.leaves()

    # 提取指定词性的单词
    def fields_leaves(self, dic_toks, grammar, pos):
        result = list()
        return {key: [i[0] for i in itertools.chain.from_iterable(self.leaves(value, grammar, pos))] for key, value in
                dic_toks.items()}

    # 过滤单词对的规则
    def acceptable_word(self, word):
        return bool(not re.match(r'[{}]'.format(punctuation), word) and not re.search(r'\d',
                                                                                      word) and word.lower() not in stopwords.words(
            'english') and 2 <= len(word) <= 40)

    # 过滤不符合规则单词、停用词、提取词干
    def filter_normalise_word(self, kwargs):
        return {key: [self.normalise(w.lower()) for w in value if self.acceptable_word(w)] for key, value in
                kwargs.items()}

    # 提取词干
    def normalise(self, word):
        lemmatizer = nltk.WordNetLemmatizer()
        word = lemmatizer.lemmatize(word)
        return word

    # 分词
    def word_from_list_tokenize(self, kwargs):
        return {key: nltk.word_tokenize("".join(value)) for key, value in kwargs.items()}


class ComputeDwpi(object):
    # 计算词频
    @staticmethod
    def term_frequency(term, document_tokens):
        counter = Counter(document_tokens)
        return counter[term] / float(len(document_tokens))

    # 计算一个文档中所有单词的词频
    @staticmethod
    def all_term_frequency(document_tokens):
        df = pd.DataFrame(data=document_tokens, columns=['单词'])
        df = df['单词'].value_counts().to_frame().reset_index()
        df.columns = ['单词', '频数']
        df['频率'] = df['频数'] / float(df['频数'].sum())
        return df

    # 计算idf
    @staticmethod
    def inverse_document_frequency(term, croups):
        count = 0
        for croup in croups:
            if term in croup:
                count += 1
        return np.log(len(croups) / float(count))

    # 计算所有单词的idf
    @classmethod
    def all_inverse_document_frequency(cls, croups):
        df_list = list()
        for croup in croups:
            df = pd.DataFrame(data=croup, columns=['单词']).drop_duplicates()
            df['idf'] = df['单词'].apply(lambda x: cls.inverse_document_frequency(x, croups))
            df_list.append(df)
            yield df

    # 计算tf-idf
    @classmethod
    def tf_idf(cls, croups):
        for df, croup in zip(cls.all_inverse_document_frequency(croups), croups):
            df_ = pd.merge(cls.all_term_frequency(croup), df, on=['单词'])
            df_['ti_idf'] = df_['频率'] * df_['idf']
            df_.sort_values(by=['ti_idf'], ascending=False, inplace=True)
            yield df_

    @staticmethod
    def term_tf_idf(term, croups):
        pass

    # 提取每个文档中的关键词到列表中
    @classmethod
    def get_keyword_list(cls, croups):
        keyword_list = list()
        for df in cls.tf_idf(croups):
            keyword_list.append(df.head(1)['单词'].values.tolist()[0])
        return keyword_list

    @classmethod
    def get_keyword_list_to_file(cls, croups, path='./keyword.csv'):
        keyword_list = cls.get_keyword_list(croups)
        with open(path, 'w+', encoding='utf-8') as f:
            f.write(" ".join(keyword_list))
        print("finally")


class CreateMatrix(object):
    # 构建一个共现矩阵
    def matrix(self, words, documents):
        # 行为关键字数，列为文档数
        matrix = np.zeros((len(words), len(documents)))
        dictionary = dict(zip(words, range(len(words))))
        for col, document in enumerate(documents):
            for word in words:
                if word in document:
                    id = dictionary[word]
                    matrix[id, col] = 1
        return matrix, dictionary

    def tf_matrix(self, words, documents):
        matrix, dictionary = self.matrix(words, documents)
        dictionary = {v: k for k, v in dictionary.items()}
        index = np.argwhere(matrix == 1)
        for i in index:
            word = dictionary[i[0]]
            matrix[i[0], i[1]] = ComputeDwpi.term_frequency(word, documents[i[1]])
        return matrix

    def idf_matrix(self, words, documents):
        # 计算IDF
        matrix, dictionary = self.matrix(words, documents)
        # 文档总数
        D = matrix.shape[1]
        # 包含每个词的文档数
        j = np.sum(matrix > 0, axis=1)
        idf = np.divide(D, j)
        return idf

    def tf_idf(self, words, documents):
        tf_matrix = self.tf_matrix(words, documents)
        idf_matrix = self.idf_matrix(words, documents)
        idf_matrix[np.isinf(idf_matrix)] = 0
        idf_matrix.reshape(len(idf_matrix), 1)
        idf_matrix = np.eye(tf_matrix.shape[0], tf_matrix.shape[0]) * idf_matrix
        tf_idf = np.dot(tf_matrix, idf_matrix)
        return tf_idf

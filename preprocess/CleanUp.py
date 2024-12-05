import re
import sys
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import pandas

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.tokenize import sent_tokenize

# from config.Globals import ReNewConfig
from config.variables import embedder, max_sentences, dim, nil, num_threads


def rm_punctuation_from_list(df_list):
    return list((re.sub(r'[^\w\s]', '', sent)) for sent in df_list)

def rm_null_from_list(df_list):
    return list(sent for sent in df_list if sent != '')

def add_null_from_list(df_list):
    len_df = len(df_list)
    for i in range(max_sentences-len_df):
        df_list.append(nil)
    return np.array(df_list,dtype=object)

def token_sentence(text):

    _sentences = sent_tokenize(text)
    return _sentences
    # Output: ['This is the first sentence.', 'Here is another one.', 'This is the third.']

def embed_single_email(sentences):
    """对单个句子进行嵌入"""
    # tmp = np.array(sentence).ravel().tolist()
    tmp = []
    for sentence_ in sentences:
        if sentence_ == nil :
            tmp.append(np.zeros(dim))
        else:
            a = embedder.encode(sentence_, show_progress_bar=False)
            # a*= 10
            tmp.append(a)

    return  tmp

def embed_batch(sentences_batch, num_threads=4):
    """使用多线程生成一批句子的嵌入"""
    # ans = []
    # 使用 ThreadPoolExecutor 并发执行编码任务
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 将句子映射到嵌入函数并行处理
        ans = list(executor.map(embed_single_email, sentences_batch))

    return ans

def renew_max_sentences(df):
    _max_sentence = 0
    for i in df:
        _max_sentence = len(i) if len(i) > _max_sentence else _max_sentence
    return _max_sentence

def shape_it(sentences):
    try:
        print("flatten list")
        if sentences is not isinstance(sentences, np.ndarray):
            sentences = np.array(sentences)
        sentences = np.vstack(sentences)

        return sentences

    except ValueError:
        print("ValueError! cannot flatten list")
        sentences = np.ravel(np.array(sentences))
        # ReNewConfig("max_sentences", renew_max_sentences(_df_list))
        sentences = np.vstack(sentences).flatten()
        return sentences

class CleanUp:
    def __init__(self, stopwords_lan="english"):
        # Download necessary NLTK resources
        # nltk.download("stopwords")
        # nltk.download('punkt_tab')
        # nltk.download("wordnet")
        self.stop_words = set(stopwords.words(stopwords_lan))
        # Lemmatization
        self.lemmatizer = WordNetLemmatizer()

    def remove_stopwords(self, text):
        words = word_tokenize(text)
        return " ".join([word for word in words if word not in self.stop_words])

    def lemmatize_text(self, text):
        words = word_tokenize(text)
        return " ".join([self.lemmatizer.lemmatize(word) for word in words])

    def cleanup(self, df: pandas.DataFrame, data_tag, batch_size=32):

        # Lowercase
        df[data_tag] = df[data_tag].str.lower()

        print("Remove numbers")
        # df[data_tag] = df[data_tag].str.replace(r"\d+", "numbers", regex=True)

        # Remove URLs
        print("Remove url")
        df[data_tag] = df[data_tag].str.replace(r"http\S+|www\S+", " urls ", regex=True)

        # Remove extra whitespace
        print("Remove extra whitespace")
        df[data_tag] = df[data_tag].str.strip()

        print("Remove stopwords")
        df[data_tag] = df[data_tag].apply(self.remove_stopwords)
        df[data_tag] = df[data_tag].apply(self.lemmatize_text)

        """
        ****************** structure change *********************
        """
        print("Remove token_sentence")
        df[data_tag] = df[data_tag].map(token_sentence)
        # Remove punctuation
        df[data_tag] = df[data_tag].map(rm_punctuation_from_list)
        # df[data_tag] = df[data_tag].map(rm_null_from_list)

        original_length = []
        for i in df[data_tag]:
            original_length.append(len(i))
        df["original_length"] = original_length

        df[data_tag] = df[data_tag].map(add_null_from_list)

        print("Batch Embedding Sentences")
        sentences = shape_it(df[data_tag])

        # sentences = np.concatenate(qwq)
        batch_embeddings = []
        for i in range(0, len(sentences), batch_size):
            sys.stdout.write(f"\r{i//batch_size + 1} / {len(sentences)//batch_size+1} is training...")
            sys.stdout.flush()
            batch = (sentences[i:i + batch_size])
            batch_embeddings.extend(embed_batch(batch, num_threads))

        # 重组嵌入回 DataFrame
        try:
            reshaped_embeddings = batch_embeddings
            """
            reshaped_elf.model =embeddings = shape_it(batch_embeddings)
            reshaped_embeddings = reshaped_embeddings.reshape(len(df),max_sentences*dim)
            df = df.drop(columns=[data_tag])
            """
            df[data_tag] = reshaped_embeddings

            pass
        except ValueError as e:
            print(f"Embedding reshape error: {e}")
            print(f"Batch embeddings: {len(batch_embeddings)}, Expected: {len(df) * max_sentences*dim}")



        return df



from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.decomposition import PCA

# 假设 data 是特征向量，labels 是对应的类别标签


class Sbert:
    def __init__(self, model='paraphrase-MiniLM-L6-v2', pca:bool=True, pca_dim=50):
        self.model = SentenceTransformer(model)
        self.pca = pca
        if self.pca:
            self.pca_model = PCA(n_components=pca_dim)

    def vectorize_df(self,df: pd.DataFrame,dataTag):
        # 将短信文本转换为列表，处理嵌入
        messages = df[dataTag].astype(str).tolist()
        embeddings = self.model.encode(messages)

        # 将嵌入转为 DataFrame，并与原始数据拼接

        if self.pca:
            embeddings = self.pca_model.fit_transform(embeddings)

        embeddings_df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])
        final_df = pd.concat([df.drop(columns=[dataTag]), embeddings_df], axis=1)
        return final_df

    def vectorize_input(self,input_sentence):
        encoded_text = self.model.encode(input_sentence)
        if self.pca:
            encoded_text = self.pca_model.transform(encoded_text)
        return encoded_text


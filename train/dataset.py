import torch
from torch import tensor
from torch.utils.data import Dataset

import numpy as np
from imblearn.over_sampling import SMOTE


class TextDataset(Dataset):
    # def __init__(self, embeddings_, labels_, max_sentences, dim, is_test=False, embedding_len = None):
    def __init__(self, embeddings_, labels_, max_sentences, dim, is_test=False):

        if not isinstance(embeddings_, np.ndarray):
            embeddings_ = np.array(embeddings_)  # Convert to NumPy array
        if not isinstance(labels_, np.ndarray):
            labels_ = np.array(labels_)

        labels_ = tensor(labels_, dtype=torch.long)

        self.labels_ = None
        self.embeddings_ = None
        """
        process embeddings_ and make it a attr
        """
        embeddings_ = np.vstack(embeddings_)
        # ****************** smote *************************
        if not is_test:
            embeddings_ = embeddings_.reshape(len(labels_), max_sentences * dim)
            smote = SMOTE(random_state=42)
            embeddings_, labels_ = smote.fit_resample(embeddings_, labels_)
            if not (0.4<np.count_nonzero(labels_)/np.size(labels_) < 0.6):
                raise ValueError(f"\n\n还是不平衡：{np.count_nonzero(labels_)/np.size(labels_)}\n\n")

        # transform into tensor
        embeddings_ = embeddings_.reshape(len(labels_), max_sentences, dim)
        embeddings_ = (tensor(embeddings_.tolist(), dtype=torch.float32))

        self.embeddings_ = embeddings_
        self.labels_ = labels_


    def __len__(self):
        return len(self.embeddings_)

    def __getitem__(self, idx):
        return self.embeddings_[idx], self.labels_[idx]


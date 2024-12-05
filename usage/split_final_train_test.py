import pandas as pd
from sklearn.model_selection import train_test_split

from config.variables import path_to_source, tag_name, text_name, path_to_final_train, path_to_final_test

df = pd.read_pickle(path_to_source)
X = df[text_name] # Feature vectors,384 dimensions
y = df[tag_name]  # Labels (e.g., spam/ham)

# 分训练集（后续还要smote 才能使用）
#returns
(X_train, X_test,
 y_train, y_test) \
    = (train_test_split(X, y, test_size=0.3, random_state=42))
pd.DataFrame({text_name:X_train,tag_name:y_train}).to_pickle(path_to_final_train)
pd.DataFrame({text_name:X_test,tag_name:y_test, }).to_pickle(path_to_final_test)

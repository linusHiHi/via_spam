import json
import torch
from sentence_transformers import SentenceTransformer
from config.path import Root_path

with open(Root_path+"/config/config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

'''
***********************config*************************************
'''
rootPath = CONFIG["root"]
path_to_source = rootPath + CONFIG["path_to"]["dataset"] + CONFIG["path_to"]["cleaned_pickle"]
path_to_k_fold = rootPath + CONFIG["path_to"]["dataset"] + CONFIG["path_to"]["k_fold_train_evaluation"]
path_to_final_train = rootPath + CONFIG["path_to"]["dataset"] + CONFIG["path_to"]["final_train"]
path_to_final_test = rootPath + CONFIG["path_to"]["dataset"] + CONFIG["path_to"]["final_test"]
path_to_statistic = rootPath + CONFIG["path_to"][ "dataset"] + CONFIG["path_to"]["statistic"]
path_to_bert_model = rootPath + CONFIG["path_to"]["bert_model"]
path_to_best_parameter = rootPath + CONFIG["path_to"]["best_parameter"]

path_to_trained_model = rootPath + CONFIG["path_to"]["model_pth"]

tag_name = CONFIG["data"]["tag_name"]
spam = CONFIG["data"]["code_name"]["spam"]
ham = CONFIG["data"]["code_name"]["ham"]
text_name = CONFIG["data"]["text_name"]

dim = CONFIG["pca_dim"] if CONFIG["pca"]=="True" else CONFIG["un_pca_dim"]


max_sentences = CONFIG["data"]["max_sentences"]
nil = CONFIG["data"]["nil"]
num_threads = CONFIG["cleanup"]["num_threads"]

original_length = CONFIG["data"]["original_length"]

try_times = CONFIG["hyper_para"]["try_times"]

# ***********************************************

input_size = dim  # SentenceTransformer embedding size

num_classes = CONFIG["hyper_para"]["num_classes"]


shuffle_train = True if CONFIG["hyper_para"]["shuffle_train"] == "True" else False
shuffle_test = False if CONFIG["hyper_para"]["shuffle_train"] == "False" else True

embedder = SentenceTransformer(path_to_bert_model)
# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import pandas as pd

from config.variables import CONFIG
from preprocess.CleanUp import CleanUp
from preprocess.txt2csv import txt2csv

txtPath = CONFIG["root"] + CONFIG["dataset"]["dir"] + CONFIG["dataset"]["original"]
sourceDataPath = CONFIG["root"] + CONFIG["dataset"]["dir"] + CONFIG["dataset"]["original_csv"]
desDataPath = CONFIG["root"] + CONFIG["dataset"]["dir"] + CONFIG["dataset"]["cleaned_pickle"]
textName = CONFIG["data"]["text_name"]
tagName = CONFIG["data"]["tag_name"]
spam = CONFIG["data"]["code_name"]["spam"]
ham = CONFIG["data"]["code_name"]["ham"]

print("start preprocessing...")
try:
    with open(sourceDataPath, 'r') as f:
        df = pd.read_csv(f)
except FileNotFoundError:
    print("source csv data not found.trying txt.")
    txt2csv(txtPath, sourceDataPath)
    with open(sourceDataPath, 'r') as f:
        df = pd.read_csv(f)

print("cleanup\n")
# cleanUp
cleanup = CleanUp()
df = cleanup.cleanup(df, textName, 64)

'''
qwqqwqwqwq
'''

df[tagName] = df[tagName].map(
    {'spam': spam, 'ham': ham})
print("write\n")

df.to_pickle(desDataPath)

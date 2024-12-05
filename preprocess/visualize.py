import pandas as pd
from matplotlib import pyplot as plt
from numpy.ma.extras import vstack
from sklearn.manifold import TSNE
from sympy.physics.units.systems.si import dimex

from config.variables import path_to_source, tag_name, text_name, ham, spam, max_sentences, dim

df = pd.read_pickle(path_to_source)

X = df[df[tag_name] == ham][text_name]
y = df[df[tag_name] == spam][text_name]

X = X.to_numpy()
X = vstack(X).reshape(len(X), max_sentences*dim)
y = y.to_numpy()
y = vstack(y).reshape(len(y), max_sentences*dim)
tsne = TSNE(n_components=2, random_state=42)
x_tsne = tsne.fit_transform(X)
y_tsne = tsne.fit_transform(y)

plt.plot(*x_tsne,label='ham')
plt.plot(*y_tsne,label='spam')
plt.legend()
plt.title("class division")
plt.show()
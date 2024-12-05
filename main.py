import json
import tkinter
from tkinter import messagebox

import numpy as np
import pandas as pd
import torch
from torch import tensor

from config.variables import path_to_best_parameter, input_size, num_classes, path_to_trained_model, tag_name, \
    text_name, max_sentences, dim, ham, spam
from preprocess.CleanUp import CleanUp
from train.conv_lstm import RNNClassifier
from train.dataset import TextDataset


class Main:
    def __init__(self):
        with open(path_to_best_parameter) as f:
            hp_params = json.load(f)

        batch_size = hp_params['batch_size']
        hidden_size = hp_params['hidden_size']
        num_epochs = hp_params['num_epochs']
        num_layers = 2
        # learning_rate = hp_params['learning_rate']
        dropout = hp_params['dropout']

        self.model = RNNClassifier(input_size, hidden_size, num_layers, batch_size, num_epochs, num_classes, dropout)
        self.model.load_state_dict(torch.load(path_to_trained_model))
        self.df = pd.DataFrame(columns=[tag_name, text_name])
        self.processed_df = None
        self.cleanup = CleanUp().cleanup
        self.map_of_ham = {spam: "spam", ham: "ham"}
    def input_email(self, raw_emails):
        if len(self.df) > 100:
            self.df = self.df.drop(self.df.index[:100])
        self.df = pd.DataFrame({tag_name: ([0]*(len(raw_emails))), text_name: raw_emails})


    def process(self):
        self.processed_df= self.cleanup(self.df,text_name,1)
        self.df = pd.DataFrame(columns=[tag_name, text_name])

    def evaluate(self):
        """ X= self.processed_df[text_name].to_numpy()
        y = self.processed_df[tag_name].to_numpy()
        dataset = TextDataset(X, y, max_sentences, dim)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
"""
        self.model.eval()
        with torch.no_grad():
            emails = self.processed_df[text_name].to_numpy()
            length = len(emails)
            emails = np.vstack(emails)
            emails = emails.reshape(length, 34, 384)
            email_tensor = tensor(emails, dtype=torch.float32)
            outputs = self.model(email_tensor)
            predicted = (outputs >= 0.5).long().tolist()
            predicted = [self.map_of_ham.get(name) for name in predicted]
            self.processed_df = None
            return predicted



if __name__ == '__main__':
    mian =Main()


    root = tkinter.Tk()
    root.title("简单ui")
    label = tkinter.Label(root, text = "请输入内容")
    label.pack(pady=10)
    entry = tkinter.Entry(root)
    entry.pack(pady=10)
    textBox = tkinter.Text(root,width=70,height=20)
    textBox.pack(pady=20)
    def show_text():
        textBox.delete(0, tkinter.END)
        for index, item in enumerate(mian.evaluate(),start=1):
            textBox.insert(tkinter.END, f"{index}:{item}\n")

    def on_click():
        emails = entry.get()

        if emails == "":
            messagebox.showwarning("input nothing")
        mian.input_email(emails)
        mian.process()


    button = tkinter.Button(root, text="submit",command=on_click)
    button.pack(pady=10)
    root.mainloop()


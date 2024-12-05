import sys

import matplotlib.pyplot as plt

import torch
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from train.utils import EarlyStopping


class RNNPlus :
    def __init__(self, model, criterion, optimizer, device,dataloader,test_dataloader,stop_delta=0.01, stop_patient=7):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.stop_delta = stop_delta
        self.stop_patient = stop_patient
        self.early_stopping = EarlyStopping(patience=self.stop_patient, delta=self.stop_delta)
        self.early_stopping_flag = False

    def epochs_t_e(self, epochs):

        train_loss_array = []
        test_loss_array = []
        # sta_glb = None
        print("\n\n"+"a new training epoch"+"\n\n")

        '''
        *************train*********************** 
        '''
        for epoch in range(epochs):
            sys.stdout.write(f"\r{epoch+1} / {epochs} is training...")
            sys.stdout.flush()

            train_loss_array.append(self.train())

            """
            ****************** evaluate for early stopping **********************
            """
            test_loss, early_stop_flag,_,_ = self.evaluate(self.test_dataloader)
            test_loss_array.append(test_loss)
            if early_stop_flag:
                return train_loss_array, test_loss_array

        """
        *****************evaluation**********************
        """
        test_loss, _, _,_ = self.evaluate(self.test_dataloader)
        test_loss_array.append(test_loss)
        fig, axs = plt.subplots(1, 2, figsize=(5, 2.7), layout='constrained')
        _, cm = plt.subplots(1, 1)
        if test_loss_array[-1] - 0.5 > 0:
            print("\n"+"error train lose"+"\n")
            axs[0].set_title("error train loss")
            axs[1].set_title("log error train loss")

        else:
            train_loss_, _, sta_glb, cm = self.evaluate(self.test_dataloader, calculate=True)
            print("\n")
            print(sta_glb)
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["spam", "ham"])


            axs[0].set_title("normal train loss")
            axs[1].set_title("log normal train loss")

        axs[0].plot(train_loss_array,label="train_loss")
        axs[0].plot(test_loss_array,label="test_loss")
        axs[0].set_xlabel('epochs')
        axs[0].set_ylabel('loss')
        axs[0].legend()

        axs[1].plot(train_loss_array,label="train_loss")
        axs[1].plot(test_loss_array,label="test_loss")
        axs[1].set_xlabel('epochs')
        axs[1].set_ylabel('loss')
        axs[1].set_yscale('log')
        axs[1].legend()
        plt.show()

        return train_loss_array, test_loss_array


    def train(self):

        #avg_train_loss_array = []
        self.model.train()
        #self.optimizer.zero_grad()
        running_train_loss = 0.0
        for batch in self.dataloader:
            embeddings, labels = batch
            embeddings, labels = embeddings.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(embeddings)
            # outputs = pad_packed_sequence(outputs,batch_first=True)

            loss = self.criterion(outputs, labels.float())
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_train_loss += loss.item()
        avg_train_loss = running_train_loss / len(self.dataloader)
        #avg_train_loss_array.append(avg_train_loss)


        return avg_train_loss

    def evaluate(self, test_dataloader,calculate=False):

        # 初始化早停对象


        self.model.eval()  # 设置模型为评估模式
        running_val_loss = 0.0
        y_true = []
        y_pred = []

        with torch.no_grad():  # 不需要计算梯度
            for batch in test_dataloader:
                embeddings, labels = batch
                embeddings, labels = embeddings.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(embeddings)  # Add seq_len dimension
                loss = self.criterion(outputs, labels.float())

                # Accumulate validation loss
                running_val_loss += loss.item()

                # Predict class labels
                predicted = (outputs >= 0.5).long()

                # Store true and predicted labels
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        avg_val_loss = running_val_loss / len(test_dataloader)
        # Generate classification report
        if calculate:
            statistic = classification_report(y_true, y_pred)
            conf_matrix = confusion_matrix(y_true, y_pred)
        else:
            statistic = None
            conf_matrix = None

        if self.early_stopping(avg_val_loss, self.model):
            print("Early stopping at epoch ")
            self.early_stopping_flag = True

        return avg_val_loss,self.early_stopping_flag,statistic,conf_matrix

    def save(self,path):
        torch.save(self.model.state_dict(),path)


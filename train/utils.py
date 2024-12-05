import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_model_wts = None
        self.delta = delta

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss + self.delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save the model's weights if it improves
            self.best_model_wts = model.state_dict()
        else:
            self.counter += 1

        # If no improvement for `patience` epochs, stop training
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop




"""
def report(y_true, y_pred):
    # Classification report
    print("Classification Report on Test Set:")
    print(classification_report(y_true, y_pred))

    # Optionally, manually calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]  # True Positive
    tn = cm[0, 0]  # True Negative
    fp = cm[0, 1]  # False Positive
    fn = cm[1, 0]  # False Negative

    # Calculate Precision, Recall, F1-Score
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # Print the results
    print(f"\nManual Calculation Metrics on Test Set:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

    return precision, recall, f1_score
"""

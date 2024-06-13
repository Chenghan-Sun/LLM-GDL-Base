import matplotlib.pyplot as plt
import torch

class EarlyStopping:
    def __init__(self, patience=10, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """ Saves model when validation loss decreases """
        self.best_loss = val_loss
        torch.save(model.state_dict(), 'checkpoint.pt')

def plot_metrics(file_name, train_losses, train_accuracies, test_accuracies):
    """ plot the loss curve and accuracies """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 6))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Train loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot train accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, 'go-', label='Train accuracy')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot test accuracy
    plt.subplot(1, 3, 3)
    plt.plot(epochs, test_accuracies, 'ro-', label='Test accuracy')
    plt.title('Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./'+file_name+'.png', dpi=300, bbox_inches='tight', pad_inches=0.1)



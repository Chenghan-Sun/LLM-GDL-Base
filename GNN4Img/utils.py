import matplotlib.pyplot as plt


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



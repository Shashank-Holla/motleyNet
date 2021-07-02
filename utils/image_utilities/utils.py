import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt



def denormalize(tensor, mean, std):
    """
    Apply denormalization on input tensor. 
    denormalized_tensor = input_tensor * Std.Dev + Mean

    Args:
        tensor (tensor): Input normalized tensor on which denormalization is to be applied. Tensor must be 4d.
        mean (tuple) : 
        std (tuple) : 
    
    Returns:
        denormalized tensor : Denormalized tensor.

    """
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')
    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    return tensor.mul(std).add(mean)

def dataset_calculate_mean_std():
        """
        Download train and test dataset, concatenate and calculate mean and standard deviation for this set.
        """
        set1 = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
        set2 = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
        data = np.concatenate([set1.data, set2.data], axis=0)
        stddev = list(np.std(data, axis=(0, 1, 2)) / 255)
        means = list(np.mean(data, axis=(0, 1, 2)) / 255)
        return stddev, means


def visualize_data(data, classes):
    i = 0
    nrows = 5
    ncols = 5
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(8,10))
    for row in range(nrows):
        for col in range(ncols):
            img, label = data[i]
            ax[row, col].imshow(img)
            ax[row, col].set_title("Label:{}".format(classes[label]))
            ax[row, col].axis("off")
            i += 1
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def draw_loss_accuracy_graph(train_loss, test_loss, train_accuracy, test_accuracy):
    fig, axs = plt.subplots(1,2,figsize=(15,6))
    axs[0].plot(train_loss, label="Train Loss")
    axs[0].plot(test_loss, label = "Test Loss")
    axs[0].set_title("Loss vs Epoch")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend(loc="best")

    axs[1].plot(train_accuracy, label="Train Accuracy")
    axs[1].plot(test_accuracy, label="Test Accuracy")
    axs[1].set_title("Accuracy vs Epoch")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy in %")
    axs[1].legend(loc="best")
    plt.figure()
    plt.show()


def capture_incorrect_classified_samples(net, device, testloader):
    """
    Captures incorrect sample data- such as labels, predictions and images
    Input
        net - model
        device - device to run the model
        testloader - testloader
    """
    net.eval()
    incorrect_labels = torch.tensor([], dtype = torch.long)
    incorrect_predictions = torch.tensor([], dtype = torch.long)
    incorrect_images = torch.tensor([])

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            
            _, predicted = torch.max(outputs.data, 1)
            result = predicted.eq(labels.view_as(predicted))

            # Incorrect labels, images and predictions           
            incorrect_labels = torch.cat((incorrect_labels,labels[~result].cpu()), dim=0)
            incorrect_predictions = torch.cat((incorrect_predictions, predicted[~result].cpu()), dim=0)
            incorrect_images = torch.cat((incorrect_images, images[~result].cpu()), dim=0)

            # Get out once we have 25 results.
            if incorrect_labels.shape[0] >=25:
                break

        return incorrect_labels.numpy(), incorrect_predictions.numpy(), incorrect_images


def show_incorrect_images(label_wrong, pred_wrong, image_wrong, classes):
    i = 0
    nrows = 5
    ncols = 5
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(12,12))
    for row in range(nrows):
        for col in range(ncols):
            ax[row, col].imshow(np.transpose((image_wrong[i].squeeze(dim=0).cpu().numpy()), (1, 2, 0)))
            ax[row, col].set_title("Predicted Label:{}\nTrue Label:{}".format(classes[pred_wrong[i]], classes[label_wrong[i]]))
            ax[row, col].axis('off')
            i += 1 
    plt.tight_layout()  
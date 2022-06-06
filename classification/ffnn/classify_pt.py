"""
Animals classification using Pytorch.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

"""
READING DATA
Using torchvision datasets
"""
# transformation to be applied to the dataset
# transforms.Resize(..),
# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32,32)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

batch_size = 32
train_folder = "/home/palma/opencampus/animals_classification_DL/dataset_pt/train"
trainset = torchvision.datasets.ImageFolder(
    root=train_folder,
    transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(
    root="/home/palma/opencampus/animals_classification_DL/dataset_pt/test",
    transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


classes = [class_name for class_name in np.sort(os.listdir(train_folder))]
print(classes)
"""
MODEL

Create the model/network
"""
# an alternative using Functional APIs
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # here define the layers
#         # the fully connected layer is nn.Linear
#         self.fc1 = nn.Linear(.., ..)
#         self.fc2 = nn.Linear(.., ..)
#         self.dp1 = nn.Dropout()
#
#     def forward(self, x):
#         # here define the forward propagation
#         # each step of the network
#
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x)) #activation function?
#
#         return x

# define the model
# examples:
# nn.Linear(),
# nn.ReLU(),
# nn.Flatten(),
# nn.Linear()
model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 15),
            nn.Sigmoid()
        )
# define loss and optimizer
loss_model = nn.CrossEntropyLoss() #.. which loss?
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # which optimizer?
print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nNumber of trainable parameters:", pytorch_total_params)
"""
TRAINING

Dedicated loop (no .fit like in tf/keras)
Forward - loss - backward propagation as we learned
"""
epochs = 5
for epoch in range(epochs):  # loop over the dataset multiple times
    print(f'epoch {epoch}')
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_model(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print(f'Finished Training\nloss: {running_loss:.3f}')

"""
EVALUATING

"""
def evaluate_model(dataloader, model):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total

acc_training, total_training = evaluate_model(trainloader, model)
print(f'Accuracy of the network on the training images: {100 * acc_training / total_training:.5f} %')
acc_test, total_test = evaluate_model(testloader, model)
print(f'Accuracy of the network on the test images: {100 * acc_test / total_test:.5f} %')

"""
Save the model
https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""
os.mkdir('models/pt_ff_basic')
torch.save(model, 'models/pt_ff_basic/pt_ff_basic.pt')
torch.save(model.state_dict(), 'models/pt_ff_basic/pt_ff_basic_state_dict.pt')

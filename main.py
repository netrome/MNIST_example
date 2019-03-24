import enum

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets
from torchvision import transforms


# Utils --------------------------------------------------------------------

Separation = enum.Enum("Separation", "TRAIN TEST")

def fashion_mnist_dataloader(separation_group: Separation) -> data.DataLoader:
    dataset = datasets.FashionMNIST(
        "~/Data/FashionMNIST/",
        train=separation_group==Separation.TRAIN,
        transform=transforms.ToTensor(),
    )

    return data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
    )

# Models -------------------------------------------------------------------

class SomethingSimple(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=2)
        self.final_conv = nn.Conv2d(16, 32, 5)

        self.fc = nn.Linear(32, 10)

    def forward(self, batch):
        x = F.relu(self.conv1(batch))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.final_conv(x))

        x = self.fc(x.view(-1, 32))
        return x

# Training loop -----------------------------------------------------------

def main():
    model = SomethingSimple()
    optimizer = torch.optim.Adam(model.parameters())

    train_loader = fashion_mnist_dataloader(Separation.TRAIN)

    epochs = 10

    criterion = nn.CrossEntropyLoss()

    loss_filter = 0
    for epoch in range(epochs):
        for iteration, (images, targets) in enumerate(train_loader):

            predictions = model(images)

            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_filter = 0.9 * loss_filter + 0.1 * float(loss)

            if (iteration + 1) % 50 == 0: 
                print(f"Iteration {iteration + 1}/{len(train_loader)} finished with loss ~ {loss_filter}", end="\r")

        print(f"Epoch {epoch} finished with loss ~ {loss_filter}")
        torch.save(model.state_dict(), 'model.params')


if __name__=="__main__":
    main()

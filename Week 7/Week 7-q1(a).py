import PIL.Image as Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import glob
from torchvision.models import AlexNet_Weights
import pandas as pd


def get_df(path, classes=['dogs', 'cats']):
    paths = pd.DataFrame({'class': [], 'path': []})
    for c in classes:
        df = pd.DataFrame({
            'class': c,
            'path': glob.glob(path + c + '/*')
        })

        paths = pd.concat([paths, df])

    paths.reset_index(inplace=False)

    return paths



class MyDataset(Dataset):
    def __init__(self, df, classes, transform=None):
        self.paths = df
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        row = self.paths.iloc[idx]
        img = Image.open(row['path'])
        if self.transform is not None:
            return self.transform(img), self.classes[row['class']]
        else:
            return img, self.classes[row['class']]



def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # L2 regularization using optimizer's weight decay
        l2_reg = sum(torch.sum(param ** 2) for param in model.parameters())
        loss += 0.001 * l2_reg  # Adjust regularization strength as needed

        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))




def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

EPOCHS = 6
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_TEST = 10
LR = 0.0001
LOG_INTERVAL = 10
RANDOM_SEED = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Resize([224, 224], antialias=False),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

CLASSES = {'dogs': 0, 'cats': 1}

torch.manual_seed(RANDOM_SEED)


train_dataset = MyDataset(get_df('cats_and_dogs_filtered/train/'), CLASSES, TRANSFORM)
test_dataset = MyDataset(get_df('cats_and_dogs_filtered/validation/'), CLASSES, TRANSFORM)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=True)

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=AlexNet_Weights.DEFAULT)
model.features.requires_grad = False

model.classifier = nn.Sequential(
    *model.classifier[:-1],
    nn.Linear(4096, 2, bias=True)
)

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
# L2 regularization using optimizer's weight decay
optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=0.001)  # Adjust weight_decay as needed


train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(EPOCHS + 1)]
for epoch in range(1, EPOCHS + 1):
    train(epoch)
    test()
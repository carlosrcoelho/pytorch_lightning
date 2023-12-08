import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import trainer

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001


# Fully connected neural network with one hidden layer
class LitNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LitNeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        images, labels = batch
        images = images.reshape(-1, 28*28)
        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        # Logging to TensorBoard by default
        self.log('train_loss', loss)

        return {"loss": loss, "log": {"train_loss": loss}}
    
    def configure_optimizers(self):        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer
    
    def train_dataloader(self):
        # MNIST dataset 
        train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                                   train=True, 
                                                   transform=transforms.ToTensor(),  
                                                   download=True)

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                   batch_size=batch_size, 
                                                   num_workers=4,
                                                   shuffle=True)
        return train_loader
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28*28)
        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        return {"val_loss": loss}
    
    def val_dataloader(self):
        # MNIST dataset 
        val_dataset = torchvision.datasets.MNIST(root='../../data', 
                                                  train=False, 
                                                  transform=transforms.ToTensor())

        # Data loader
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                                  batch_size=batch_size, 
                                                  num_workers=4,
                                                  shuffle=False)
        return val_loader
    
    def on_validation_epoch_end(self):
        return super().on_validation_epoch_end()
    

if __name__ == '__main__':
    trainer = pl.Trainer(max_epochs=num_epochs, fast_dev_run=False, deterministic=True)
    model = LitNeuralNet(input_size, hidden_size, num_classes)
    trainer.fit(model)

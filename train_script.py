import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import multiprocessing as mp
import torchvision
from torchvision import transforms
from torch.utils.data import random_split

STUDY_NAME = 'fmnist_study'
STORAGE_PATH = f'sqlite:///{STUDY_NAME}.db'

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

train_val_split = 0.8

# This dataset is already "sorted" as part of the import method, but no "validation" set has been selected in this case
# Loading the FashionMNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Calculate sizes
train_size = int(train_val_split * len(train_dataset))
val_size = len(train_dataset) - train_size

# Split the dataset
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# Here we define the model parameters -- the general strucutre as provided here will produce a fully connected network [28x28] --> 32 --> 16 --> 10
class MLP(nn.Module): # MLP stands for "Multi-Layer Perceptron"
    def __init__(self, layer_sizes): # this initializes the structure of the network
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drops = nn.ModuleList()
        self.ELU = nn.ELU()
        input_size = 28 * 28 ## 28*28 input features and 32 outputs
        for size in layer_sizes:
            self.layers.append(nn.Linear(input_size, size))
            self.norms.append(nn.BatchNorm1d(size))
            self.drops.append(nn.Dropout(0.2))
            input_size = size
        
        self.output_layer = nn.Linear(input_size, 10) ## 10 output features because MNIST has 10 target classes

    def forward(self, x): # this modifies the elements of the intial structure defined above
        x = x.view(-1, 28 * 28)  # Flatten the input
        for layer, norm, drop in zip(self.layers, self.norms, self.drops):
            x = self.ELU(norm(layer(x))) # batch normalization before activation
            x = drop(x) # dropout after activation
        x = self.output_layer(x)
        return x

def objective(trial):
    optimizers = {
        'SGD': optim.SGD,
        'Adam': optim.Adam,
        'RMSprop': optim.RMSprop,
    }

    layer_size = trial.suggest_categorical(f'layer_size_0', [16, 32, 64, 128, 256, 512])

    # Suggest optimizer
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'RMSprop'])

    # Suggest batch size
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # Initialize the model
    model = MLP([layer_size])
    criterion = nn.CrossEntropyLoss()

    # Create optimizer
    optimizer_class = optimizers[optimizer_name]
    optimizer = optimizer_class(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Data loaders with suggested batch size
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation at the end of each epoch
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total

        # Report intermediate objective value to the trial
        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

def run_optuna():
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_PATH)
    study.optimize(objective, n_trials=4)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    reduction_factor = 4 # based on desired cpu load / available cpus
    num_workers = int(mp.cpu_count() / reduction_factor)
    print(f'Number of workers: {num_workers}')
    processes = []
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        load_if_exists=True,
        direction='maximize',
        pruner=pruner
    )
    # run N trials for each cpu core
    for _ in range(num_workers):
        p = mp.Process(target=run_optuna)
        print('Starting process')
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print('Best hyperparameters:', study.best_params)
    print('Best accuracy:', study.best_value)
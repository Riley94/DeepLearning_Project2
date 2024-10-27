import optuna
from optuna.integration import KerasPruningCallback
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import to_categorical
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split

STUDY_NAME = 'fmnist_study_cnn'
STORAGE_PATH = f'sqlite:///{STUDY_NAME}.db'

# Load and preprocess the data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42)

def objective(trial):
    # Hyperparameter suggestions
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 3)
    num_filters = []
    for i in range(num_conv_layers):
        num_filters.append(trial.suggest_categorical(f'num_filters_{i}', [16, 32, 64, 128]))
    # Fully connected layer sizes
    fc_unit = trial.suggest_categorical(f'fc_units', [64, 128, 256])
    # Activation function
    activation = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh'])
    # Optimizer
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'RMSprop'])
    # Learning rate
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    # Number of epochs
    epochs = 10
    # Dropout rate
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    # Build the model
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(num_filters[0], kernel_size=(3, 3),
                     activation=activation, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    for i in range(1, num_conv_layers):
        model.add(Conv2D(num_filters[i], kernel_size=(3, 3),
                         activation=activation, padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(fc_unit, activation=activation))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    # Compile the model
    if optimizer_name == 'SGD':
        optimizer = SGD(learning_rate=lr)
    elif optimizer_name == 'Adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = RMSprop(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    keras_pruning_callback = KerasPruningCallback(trial, monitor='val_accuracy')

    # Train the model
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_data=(x_val, y_val),
                        callbacks=[keras_pruning_callback],
                        verbose=0)
    # Evaluate the model
    score = model.evaluate(x_val, y_val, verbose=0)
    val_accuracy = score[1]
    
    # Report the validation accuracy to Optuna
    return val_accuracy

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
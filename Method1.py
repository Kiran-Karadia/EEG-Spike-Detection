import shared_functions as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix

# Preprocessing data for MLP
def prepare_MLP_data(data, indexes, labels, split_percentage):
    # data - Filtered, normalised data
    # indexes - indexes of spikes
    # labels - labels of each spike
    # split_percentage - Train/test percentage split

    all_spikes = sf.get_all_spikes(data, indexes, samps_back=5, samps_forward=40)
    split_point = round(len(all_spikes) * split_percentage)

    train_data = all_spikes[:split_point]       # Split data for training
    train_labels = labels[:split_point]         # Split labels for training

    test_data = all_spikes[split_point:]        # Split data for testing
    test_labels = labels[split_point:]          # Split labels for testing

    return train_data, train_labels, test_data, test_labels

# Train MLP model
def MLP_train_model(input, labels):
    clf = MLPClassifier(hidden_layer_sizes=(20, 10),
                        activation='tanh',
                        alpha=0.0001,
                        solver='adam',
                        learning_rate='constant',
                        max_iter=1000,
                        shuffle=True,
                        random_state=1,
                        verbose=False)
    clf.fit(input, labels)
    return clf

# Makes predictions on submission data and display results
def prediction_and_stats(model, data, sample_offset):
    print("------------Now using the submission data------------")
    bpf = sf.bandpass_filter(25e3, 300, 1500, 3)
    # Filter the data
    data = signal.sosfiltfilt(bpf, data)
    # Search for spikes
    indexes = sf.find_spike_indexes(data, coeff=1.4, search=3)
    print("Number of spikes found in submission data:", len(indexes))
    print("Remove indexes that are too close together")

    # Remove spikes that are within 45 samples of each other
    new_indexes = []
    new_indexes.append(indexes[0])
    i = 0;
    while i < len(indexes)-1:
        diff = indexes[i+1] - new_indexes[-1]
        if diff > 45:
            new_indexes.append(indexes[i+1])
        i += 1
    indexes = np.array(new_indexes)

    # Add sample offset
    indexes = indexes + sample_offset
    print("Number of spikes in submission data:", len(indexes))
    # Normalise the data
    data = sf.normalise(data)
    all_spikes = sf.get_all_spikes(data, indexes, samps_back=5, samps_forward=40)

    # Predict using the model
    predictions = model.predict(all_spikes)


    ones = np.count_nonzero(predictions == 1)
    twos = np.count_nonzero(predictions == 2)
    threes = np.count_nonzero(predictions == 3)
    fours = np.count_nonzero(predictions == 4)

    print("Class 1 was predicted", ones,  "times.")
    print("Class 2 was predicted", twos, "times.")
    print("Class 3 was predicted", threes, "times.")
    print("Class 4 was predicted", fours, "times.")
    return

# Create confusion matrix plot
def disp_confusion_matrix(data, labels, title):
    disp = plot_confusion_matrix(model, data, labels,
                                 cmap=plt.cm.Blues)
    disp.ax_.set_title(title)
    return

# Search for optimal hyperparameters
def optimise_mlp(data, labels):
    mlp = MLPClassifier(max_iter=1000)
    parameter_space = {
        'hidden_layer_sizes': [(64, 32, 16, 8),
                               (20, 10, 5),
                               (20, 5),
                               (20, 10)],
        'activation': ['tanh', 'logistic', 'relu', 'identity'],
        'solver': ['adam', 'lbfgs', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'adaptive'],
    }
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, verbose=1, refit=True)
    clf.fit(data, labels)
    # Best parameters set
    print('Best parameters were:\n', clf.best_params_)
    model = MLPClassifier(**clf.best_params_)
    model.fit(data, labels)
    return model


# Load training data, indexes, classes
d_train_raw, index_train, class_train = sf.load_data('Datasets/trainDataClean.mat')
# Load submission data
d_sub = sf.load_data('Datasets/submission.mat')

# Create bandpass filter
bpf = sf.bandpass_filter(25e3, 200, 1500, 3)
# Filter the raw data
d_train_filt = signal.sosfiltfilt(bpf, d_train_raw)
# Normalise the filtered data
d_filt_norm = sf.normalise(d_train_filt)

# Prepare data for training the MLP - Filter->Normalise->Split
[train_data,
 train_labels,
 test_data,
 test_labels] = prepare_MLP_data(data=d_filt_norm, indexes=index_train, labels=class_train, split_percentage=0.2)

# Train MLP model with training data and training labels
model = MLP_train_model(input=train_data, labels=train_labels)

# Optimise hyperparameters
#model = optimise_mlp(train_data, train_labels)

# Test model with test data and print the score
print("The model has a score of:", model.score(test_data, test_labels), "with the labelled test data.\n")
disp_confusion_matrix(test_data, test_labels, 'Test indexes and labels')


# Search for spikes in training data
indexes = sf.find_spike_indexes(data=d_train_filt, coeff=0.5, search=3)
print(len(indexes), "spikes were found with the spike detector. There are actually", len(index_train), "spikes.\n")

# Get indexes that fall within +/- 50 samples of the real indexes
new_index, new_label, sample_offset = sf.find_correct_indexes(indexes, index_train, class_train)

# Get spikes with found indexes (with sample offset)
new_spikes = sf.get_all_spikes(d_filt_norm, indexes=new_index, samps_back=5, samps_forward=40)

# Test model with found spikes
new_score = model.score(new_spikes, new_label)
print("The model has a score of:", new_score, "with spikes found using the spike detector.\n")

prediction_and_stats(model, d_sub, sample_offset)

plt.show()
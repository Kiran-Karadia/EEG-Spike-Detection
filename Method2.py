import shared_functions as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

def display_clusters(data, kmean, fignum, title):
    plt.figure(fignum)
    # Step size of the mesh
    h = .05

    # Plot the decision boundary. Assign a colour to each
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmean.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.clf()
    plt.imshow(Z, interpolation="nearest",
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired, aspect="auto", origin="lower")

    plt.plot(data[:, 0], data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmean.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
                color="w", zorder=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    return

def pca_data(data, indexes, components):
    # Get all spikes with given indexes
    all_spikes = sf.get_all_spikes(data=data, indexes=indexes, samps_back=5, samps_forward=40)
    # Reduce data
    reduced_data = PCA(n_components=components).fit_transform(all_spikes)
    return reduced_data

def find_cluster_mappings(predictions, labels):
    # Get array of indexes for each cluster
    zeros = np.where(predictions == 0)
    ones = np.where(predictions == 1)
    twos = np.where(predictions == 2)
    threes = np.where(predictions == 3)

    # Find mean value of respective indexes in labels
    mappings = [round(np.mean(labels[zeros])),
                round(np.mean(labels[ones])),
                round(np.mean(labels[twos])),
                round(np.mean(labels[threes]))]
    return mappings

def update_predictions(predictions, mappings):
    # Loop through each prediction and map predicted cluster to class
    for i in range(len(predictions)):
        if predictions[i] == 0:
            predictions[i] = mappings[0]

        elif predictions[i] == 1:
            predictions[i] = mappings[1]

        elif predictions[i] == 2:
            predictions[i] = mappings[2]

        elif predictions[i] == 3:
            predictions[i] = mappings[3]
    return predictions

def train_test_split(r_data, labels, split_percentage):
    # Find split point for train and test
    split_point = round(len(r_data) * split_percentage)

    # Split data into train and test sets
    reduced_train = r_data[:split_point]
    train_labels = labels[:split_point]

    reduced_test = r_data[split_point:]
    test_labels = labels[split_point:]
    return reduced_train, train_labels, reduced_test, test_labels

def optimise_kmean(data):
    # Optimise parameters
    kmean = KMeans(n_clusters=4)
    # Possible paramaters
    parameter_space = {'n_clusters': [4],
                        'init': ['k-means++', 'random'],
                        'n_init': [10, 20, 30, 40, 50],
                        'algorithm': ['auto', 'full', 'elkan'],
    }
    # Performe Grid search
    optimised_model = GridSearchCV(kmean, parameter_space, n_jobs=-1, verbose=1, refit=True)
    optimised_model.fit(data)
    # Best parameters set
    print('Best parameters were:\n', optimised_model.best_params_)
    # Set best parameters
    model = KMeans(**optimised_model.best_params_)
    return model

def train_kmean(data, indexes, classes, components):
    # Filter the data
    data = signal.sosfiltfilt(bpf, data)
    # Scale the data
    data = scale(data)

    # Reduce all spikes
    reduced_all_spikes = pca_data(data, indexes=indexes, components=components)

    [reduced_train,
     train_labels,
     reduced_test,
     test_labels] = train_test_split(r_data=reduced_all_spikes, labels=classes, split_percentage=0.8)

    # Create Kmeans
    kmean = KMeans(n_clusters=4)
    # Fit Kmeans model
    kmean.fit(reduced_train)
    # Optimise parameters
    #kmean = optimise_kmean(reduced_all_spikes)
    # Fit with optimised paramters
    kmean.fit(reduced_train)

    # Display clusters
    display_clusters(reduced_train, kmean, 1, "K-means Clusters")

    # Reduce test data using PCA
    predictions = kmean.predict(reduced_test)

    # Find what class each cluster should be mapped to
    mappings = find_cluster_mappings(predictions=predictions, labels=test_labels)
    # Edit predictions with mappings
    predictions = update_predictions(predictions=predictions, mappings=mappings)

    # Calculate and print score
    score = len(predictions[predictions==test_labels]) / len(predictions)
    print("PCA components =", components, "Labelled test data accuracy:", score)
    print("Silhouette score:", silhouette_score(reduced_test, predictions, metric='euclidean'))

    return kmean, mappings

# Load training data, indexes, classes
d_train_raw, index_train, class_train = sf.load_data('Datasets/trainDataClean.mat')
# Load submission data
d_sub = sf.load_data('Datasets/submission.mat')

# Create bandpass filter
bpf = sf.bandpass_filter(25e3, 300, 1500, 3)
# Filter the raw data
d_train_filt = signal.sosfiltfilt(bpf, d_train_raw)
# Scale the filtered data
d_filt_scale = scale(d_train_filt)


# Create model and get cluster -> label mappings
kmean, mappings = train_kmean(d_train_raw, index_train, class_train, components=2)


# Search for spikes in training data
print("Searching for spikes in trained data using spike detector.")
indexes = sf.find_spike_indexes(data=d_train_filt, coeff=1.4, search=3)
print(len(indexes), "spikes were found with the spike detector. There are actually", len(index_train), "spikes.")

# Get indexes that fall within +/- 50 samples of the real indexes
new_index, new_label, sample_offset = sf.find_correct_indexes(found_indexes=indexes, real_indexes=index_train, real_labels=class_train)


# Reduce new found spikes
reduce_new_spikes = pca_data(data=d_filt_scale, indexes=new_index, components=2)
# Predict with model
predictions = kmean.predict(reduce_new_spikes)
# Edit predictions with mappings
predictions = update_predictions(predictions=predictions, mappings=mappings)

# Calculate and print score
score = len(predictions[predictions == new_label]) / len(predictions)
print("Found spikes score:", score)


print("------------Now using the submission data------------")
d_sub_filt = signal.sosfiltfilt(bpf, d_sub)
d_sub_scale = scale(d_sub_filt)

# Search for spikes in submission data
indexes = sf.find_spike_indexes(d_sub_filt, coeff=1.4, search=3)
print("Number of spikes found:", len(indexes))
print("Remove indexes that are too close together")
new_indexes = []
new_indexes.append(indexes[0])
i = 0;
while i < len(indexes) - 1:
    diff = indexes[i + 1] - new_indexes[-1]
    if diff > 45:
        new_indexes.append(indexes[i + 1])
    i += 1
indexes = np.array(new_indexes)
print("Number of spikes in submission data:", len(indexes))

# Reduce the found spikes
reduced_spikes = pca_data(data=d_sub_scale, indexes=indexes, components=2)

# Make predictions
sub_predictions = kmean.predict(reduced_spikes)
# Update labels
sub_predictions = update_predictions(predictions=sub_predictions, mappings=mappings)

ones = np.count_nonzero(sub_predictions == 1)
twos = np.count_nonzero(sub_predictions == 2)
threes = np.count_nonzero(sub_predictions == 3)
fours = np.count_nonzero(sub_predictions == 4)

print("Class 1 was predicted", ones, "times.")
print("Class 2 was predicted", twos, "times.")
print("Class 3 was predicted", threes, "times.")
print("Class 4 was predicted", fours, "times.")

display_clusters(reduced_spikes, kmean, 2, 'submission')
print("Silhouette score:", silhouette_score(reduced_spikes, sub_predictions, metric='euclidean'))

plt.show()



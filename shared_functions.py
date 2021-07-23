import scipy.io as spio
import numpy as np
from scipy import signal
from sklearn.preprocessing import MinMaxScaler

# Load a .mat file from a given path
def load_data(file_path):
    # Load the .mat file
    mat = spio.loadmat(file_path, squeeze_me=True)
    d = mat['d']    # Get data row
    if len(mat) == 6:   # Get Index and Class row if present
        Index = mat['Index']
        Class = mat['Class']
        return np.array(d), np.array(Index), np.array(Class)
    return np.array(d)

# Create a bandpass filter
def bandpass_filter(fs, low_cutoff, high_cutoff, order):
    # fs    - Sampling frequency
    # low_cutoff    - Lower cut-off frequency
    # high_cutoff   - Higher cut-off frequency
    # order         - Order number of the filter
    nyq = fs / 2                # Nyquist frequency
    low = low_cutoff / nyq      # Normalise from 0-1, where 1 = nyq
    high = high_cutoff / nyq    # Normalise from 0-1, where 1 = nyq
    sos = signal.butter(order,
                        [low, high],
                        btype='bandpass',   # Type of filter
                        analog=False,       # Digital filter
                        output='sos')       # Second-order section representation
    # (Padding is symmetric by default)
    return sos

# Normalise data between 0-1
def normalise(data):
    scaler = MinMaxScaler()           # Create the scaler for normalisation
    data = np.reshape(data, [-1, 1])                  # Reshape data (2D array required for MinMaxScaler)
    data_norm = scaler.fit_transform(data)      # Fit and transform data
    data_norm = data_norm.flatten()             # Flatten back to 1D
    return data_norm

# Find indexes of spikes
def find_spike_indexes(data, coeff, search):
    # data  - The signal/wave to search for spikes
    # coeff - Coefficient for calculating threshold
    # search    - The number of samples to look ahead when searching for a spike
    sigma = np.std(data)    # Standard deviation of the data
    threshold = coeff*sigma # Height threshold to determine if currently in a spike
    spike_indexes = []
    spike_waveform = []
    # Loop through each datapoint (end before search exceeds length of data)
    for i in range(len(data)-search-1):
        spike_waveform.append(0)
        # If current point is above the threshold, assume to be in a spike
        if data[i+search] >= threshold:
            j = 0
            # While each point up, to search limit, is increasing
            while (data[i+j] > data[i+j-1]) and j <= search:
                # If search limit has been reached
                if j == search:
                    spike_waveform[i] = 1
                    # If previous point is not a spike, current point is start of a spike
                    if spike_waveform[i-1] == 0:
                        # Store the index
                        spike_indexes.append(i)
                j += 1
    spike_waveform = np.array(spike_waveform)
    return np.array(spike_indexes)

# Store each spike
def get_all_spikes(data, indexes, samps_back, samps_forward):
    # data      - The signal/wave to get spikes from
    # indexes   - The indexes of each spike
    # samps_back- number of samples to go back from index
    # samps_forward - Number of samples to go forward from index

    spike_length = samps_back+samps_forward
    all_spikes = np.empty([len(indexes), (samps_back+samps_forward)])   # Allocate space for spikes
    # Loop through each index
    for i in range(len(indexes)):
        # Get star point of spike
        start = indexes[i] - samps_back
        # Get end point of spike
        end = indexes[i] + samps_forward
        if end <= len(data) and start >= 0:
            # Get entire spike
            all_spikes[i] = data[start:end]
    return all_spikes

# Cleanup found indexes
def find_correct_indexes(found_indexes, real_indexes, real_labels):
    # Cross-reference indexes with real indexes
    # Loops through both index array and only appends when each element is within
    # 45 samples of each other
    new_indexes = []
    new_labels = []
    total_diffs = 0
    j = 0
    i = 0
    # Cross-reference indexes with real indexes

    while i < len(found_indexes) and j < len(real_indexes):
        diff = abs(found_indexes[i] - real_indexes[j])
        total_diffs = total_diffs + found_indexes[i] - real_indexes[j]
        if diff <= 45:
            new_indexes.append(found_indexes[i])
            new_labels.append(real_labels[j])
            i += 1
            j += 1
        elif found_indexes[i] > real_indexes[j]:
            j += 1
        elif found_indexes[i] < real_indexes[j]:
            i += 1

    new_indexes = np.array(new_indexes)
    new_labels = np.array(new_labels)

    # Calculate average sample difference between real and found indexes
    sample_offset = int(np.ceil(abs(np.mean(new_indexes - real_indexes[:len(new_indexes)]))))

    # New array that has only has indexes that are known to be near a spike
    new_indexes = new_indexes + sample_offset
    return new_indexes, new_labels, sample_offset

def save_to_mat(index, labels, file_name):
    spio.savemat(file_name, {'Index': index, 'Class': labels})


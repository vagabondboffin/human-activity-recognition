import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def load_dataset():
    file_path = 'A:\py\pythonProjects\HAR\datasets\WISDM\WISDM_ar_v1.1_raw.txt'

    columns = ['user', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip().replace(';', '')  # Remove trailing ';' and any leading/trailing whitespace
                parts = line.split(',')
                if len(parts) == 6:
                    data.append(parts)
    except Exception as e:
        print(f"Error reading data file: {e}")
        return None

    data = pd.DataFrame(data, columns=columns)

    # Convert numeric columns to appropriate data types
    numeric_cols = ['user', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop rows with missing values
    data = data.dropna()

    count_of_activity = data['activity'].value_counts()
    print(count_of_activity)

    return data

def create_feature_sequences(data, seq_length=100):
    sequences = []
    for i in range(0, len(data) - seq_length + 1, seq_length):
        sequence = data[i:i + seq_length]
        sequences.append(sequence)
    return sequences

def create_label_sequences(labels, seq_length=100):
    sequence_labels = []
    for i in range(0, len(labels) - seq_length + 1, seq_length):
        sequence = labels[i:i + seq_length]
        most_frequent_label = np.bincount(sequence).argmax()
        sequence_labels.append(most_frequent_label)
    return sequence_labels

def preprocess_data(df, seq_length=100):
    df.sort_values(['user', 'timestamp'], inplace=True)

    label_encoder = LabelEncoder()
    df['activity'] = label_encoder.fit_transform(df['activity'])

    X_features = ['x-axis', 'y-axis', 'z-axis']
    X = df[X_features].values
    y = df['activity'].values

    X_3d = create_feature_sequences(X, seq_length=seq_length)
    y_seq = create_label_sequences(y, seq_length=seq_length)

    X = np.array(X_3d)
    y = np.array(y_seq)

    return X, y, label_encoder

def one_hot_encode_labels(y):
    onehot_encoder = OneHotEncoder(categories='auto')
    y_onehot = onehot_encoder.fit_transform(y.reshape(-1, 1)).toarray()  # Convert to dense array
    return y_onehot

# Example usage
if __name__ == "__main__":
    df = load_dataset()
    X, y, label_encoder = preprocess_data(df)
    y_onehot = one_hot_encode_labels(y)

# print('h')

import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

from keras.callbacks import TensorBoard
from datasets.wisdmLoader import load_dataset, preprocess_data, one_hot_encode_labels
from classificationModel.model import multihead_model

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, recall_score, f1_score
from scipy import stats as st
from keras.backend import clear_session

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

tensorboard_callback = TensorBoard(log_dir="./logs")

NUM_FOLDS = 3
EPOCH = 50
BATCH_SIZE = 16
LSTM_UNITS = 32
CNN_FILTERS = 3
NUM_LSTM_LAYERS = 1
LEARNING_RATE = 1e-4
PATIENCE = 20
SEED = 0
F = 64
D = 12
results = []


MODE = ''
SAVE_DIR = './model_with_self_attn_' + MODE + '_results'

# Load and preprocess the data
df = load_dataset()
X, y, label_encoder = preprocess_data(df)
y_one_hot = one_hot_encode_labels(y)

NUM_LABELS = y_one_hot.shape[1]

avg_acc = []
avg_recall = []
avg_f1 = []
early_stopping_epoch_list = []
# Define the number of splits for cross-validation
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

# Example usage of cross-validation
BATCH_SIZE = 32

for fold_index, (train_idx, test_idx) in enumerate(kf.split(X)):

    X_train, y_train, y_train_one_hot = X[train_idx], y[train_idx], y_one_hot[train_idx]
    X_test, y_test, y_test_one_hot = X[test_idx], y[test_idx], y_one_hot[test_idx]

    X_train_ = np.expand_dims(X_train, axis=3)
    X_test_ = np.expand_dims(X_test, axis=3)

    train_trailing_samples = X_train_.shape[0] % BATCH_SIZE
    test_trailing_samples = X_test_.shape[0] % BATCH_SIZE

    if train_trailing_samples != 0:
        X_train_ = X_train_[: -train_trailing_samples]
        y_train_one_hot = y_train_one_hot[: -train_trailing_samples]
        y_train = y_train[: -train_trailing_samples]
    if test_trailing_samples != 0:
        X_test_ = X_test_[: -test_trailing_samples]
        y_test_one_hot = y_test_one_hot[: -test_trailing_samples]
        y_test = y_test[: -test_trailing_samples]
    # Ensure data types are float32
    X_train_ = X_train_.astype('float32')
    y_train_one_hot = y_train_one_hot.astype('float32')
    X_test_ = X_test_.astype('float32')
    y_test_one_hot = y_test_one_hot.astype('float32')

    print(y_train.shape, y_test.shape)

        # Define and compile the model with fixed hyperparameters
    rnn_model = multihead_model(x_train=X_train_, num_labels=NUM_LABELS, LSTM_units=LSTM_UNITS,
                                  num_conv_filters=CNN_FILTERS, batch_size=BATCH_SIZE, F=F, D=D, num_heads=8)

    model_filename = SAVE_DIR + '/best_model_with_self_attn_' + 'WISDM' + '_fold_' + str(
        fold_index) + '_F_' + str(F) + '_D_' + str(D) + '.weights.h5'
    callbacks = [ModelCheckpoint(filepath=model_filename, monitor='val_accuracy', save_weights_only=True,
                                  save_best_only=True), EarlyStopping(monitor='val_accuracy',
                                                                     patience=PATIENCE)]

    opt = optimizers.Adam(clipnorm=1.)

    rnn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = rnn_model.fit(X_train_, y_train_one_hot, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1,
                            callbacks=[callbacks, tensorboard_callback], validation_data=(X_test_, y_test_one_hot))
    early_stopping_callback = callbacks[1]

    early_stopping_epoch = early_stopping_callback.stopped_epoch - PATIENCE + 1
    print('Early stopping epoch: ' + str(early_stopping_epoch))
    early_stopping_epoch_list.append(early_stopping_epoch)

    rnn_model.save_weights(model_filename)

    if early_stopping_epoch <= 0:
        early_stopping_epoch = -100

    # Evaluate model and predict data on TEST
    print("******Evaluating TEST set*********")
    rnn_model.load_weights(model_filename)

    y_test_predict = rnn_model.predict(X_test_, batch_size=BATCH_SIZE)
    y_test_predict = np.argmax(y_test_predict, axis=1)

    acc_fold = accuracy_score(y_test, y_test_predict)
    avg_acc.append(acc_fold)

    recall_fold = recall_score(y_test, y_test_predict, average='macro')
    avg_recall.append(recall_fold)

    f1_fold = f1_score(y_test, y_test_predict, average='macro')
    avg_f1.append(f1_fold)

    print('Accuracy[{:.4f}] Recall[{:.4f}] F1[{:.4f}] at fold[{}]'.format(acc_fold, recall_fold, f1_fold,
                                                                           fold_index))
    print('______________________________________________________')
    clear_session()

    # Store the results
    results.append({'params': {'F': F, 'D': D}, 'accuracy': acc_fold, 'recall': recall_fold, 'f1_score': f1_fold})

ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
ic_recall = st.t.interval(0.9, len(avg_recall) - 1, loc=np.mean(avg_recall), scale=st.sem(avg_recall))
ic_f1 = st.t.interval(0.9, len(avg_f1) - 1, loc=np.mean(avg_f1), scale=st.sem(avg_f1))

print('Mean Accuracy[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))
print('Mean Recall[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_recall), ic_recall[0], ic_recall[1]))
print('Mean F1[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_f1), ic_f1[0], ic_f1[1]))


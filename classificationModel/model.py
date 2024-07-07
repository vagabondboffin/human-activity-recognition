from tensorflow.keras.layers import Input, Conv2D, LSTM, Dense, Lambda, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from classificationModel.self_attention import SelfAttention  # Assuming this is a custom module you have

def multihead_model(x_train, num_labels, LSTM_units, num_conv_filters, batch_size, F, D, num_heads):

    cnn_inputs = Input(shape=(x_train.shape[1], x_train.shape[2], 1), batch_size=batch_size, name='rnn_inputs')
    cnn_layer = Conv2D(num_conv_filters, kernel_size=(1, x_train.shape[2]), strides=(1, 1), padding='valid', data_format="channels_last")
    cnn_out = cnn_layer(cnn_inputs)

    # Lambda layer to squeeze the output
    sq_layer = Lambda(lambda x: K.squeeze(x, axis=2), output_shape=(x_train.shape[1], num_conv_filters))
    sq_layer_out = sq_layer(cnn_out)

    rnn_layer = LSTM(LSTM_units, return_sequences=True, name='lstm', return_state=True)
    rnn_layer_output, _, _ = rnn_layer(sq_layer_out)

    # Add multiple self-attention heads
    attention_outputs = []
    for _ in range(num_heads):
        attention_output, _ = SelfAttention(size=F, num_hops=D, use_penalization=False, batch_size=batch_size)(rnn_layer_output)
        attention_outputs.append(attention_output)

    # Concatenate the outputs of multiple self-attention heads
    if num_heads > 1:
        encoder_output = Concatenate(axis=-1)(attention_outputs)
    else:
        encoder_output = attention_outputs[0]

    dense_layer = Dense(num_labels, activation='softmax')
    dense_layer_output = dense_layer(encoder_output)

    model = Model(inputs=cnn_inputs, outputs=dense_layer_output)
    print(model.summary())

    return model

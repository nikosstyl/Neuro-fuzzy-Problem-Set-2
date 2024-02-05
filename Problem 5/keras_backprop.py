import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable all debugging logs
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
USE_TF = True
if USE_TF:
    from tensorflow import keras
else:
    import keras

from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Dense
from keras.initializers import RandomUniform


# Define the target function
def target_function(p):
    return 1 + np.sin(3 * np.pi * p / 8)

# Define the model function
def build_model(hidden_units, learning_rate, dropout_rate):
    model = keras.Sequential()
    
    first_layer = Dense(hidden_units, input_dim=1, activation='sigmoid',
                                  kernel_initializer=RandomUniform(-0.5, 0.5),
                                  bias_initializer=RandomUniform(-0.5, 0.5))
    middle_layer = Dropout(dropout_rate)
    second_layer = Dense(1, activation='relu', input_dim=hidden_units,
                                  kernel_initializer=RandomUniform(-0.5, 0.5),
                                  bias_initializer=RandomUniform(-0.5, 0.5))
    
    model.add(first_layer)
    model.add(middle_layer)
    model.add(second_layer)
    
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def backpropagation(export_figs):
    # Generate training data
    p_train = np.linspace(-2, 2, 300)
    g_train = target_function(p_train)

    # Training configurations
    hidden_unit = 12
    lr = 0.1
    # dropout_rates = [0.15, 0.25, 0.35]
    dropout_rates = [0.15]

    MAX_EPOCHS = 10000

    early_stopping = EarlyStopping(monitor='loss', patience=120)

    # Experiment with different configurations
    for dropout in dropout_rates:
        print(f"\nTraining with dropout rate: {dropout}\n")

        model = build_model(hidden_unit, lr, dropout)

        # Train the model
        history = model.fit(p_train, g_train, epochs=MAX_EPOCHS, batch_size=1, verbose=1, callbacks=[early_stopping], use_multiprocessing=True)
        
        # Plot the results
        # plt.figure()
        f, (ax1,ax2) = plt.subplots(2,1, layout='constrained')

        ax1.plot(history.history['loss'])
        ax1.set_title(f'Error over epochs')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        # Predict and plot the function approximation
        p_test = np.linspace(-2, 2, 100)
        g_pred = model.predict(p_test)
        
        # plt.figure()
        ax2.plot(p_test, g_pred, '-', label=f'Predicted')
        ax2.plot(p_test, target_function(p_test), '--', label='Target')
        ax2.legend()
        ax2.set_title(f'Function Approximation with dropout rate = {dropout}')
        ax2.set_xlabel('p')
        ax2.set_ylabel('g(p)')
        ax2.grid(True)

        if export_figs:
            plt.savefig(f'nn_1_12_1_{dropout}.pdf', format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print(f'{sys.argv[0]}: error: Invalid number of arguments')
        print(f'\tUsage: python3 {sys.argv[0]} (optional: export_figs=true/false)')
        exit(1)
    
    for i in range(len(sys.argv)):
        sys.argv[i] = sys.argv[i].lower()

    export_figs = False

    if 'export_figs=true' in sys.argv:
        export_figs = True
    
    backpropagation(export_figs)
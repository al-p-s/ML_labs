import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

df0 = pd.read_csv('../z_datasets/nn_0.csv')
df1 = pd.read_csv('../z_datasets/nn_1.csv')


def prepare(df):
    X = df[['X1', 'X2']].values
    y = ((df['class'] + 1) / 2).values  # -1/1 → 0/1
    return X, y

X0, y0 = prepare(df0)
X1, y1 = prepare(df1)

# Обучение на nn_0
results = {}
for activation in ['sigmoid', 'tanh']:
    for optimizer in ['adam', 'sgd']:
        model = keras.Sequential([layers.Dense(1, activation=activation, input_shape=(2,))])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(X0, y0, epochs=200, verbose=0)
        final_acc = history.history['accuracy'][-1]
        epochs_to_90 = next((i for i, a in enumerate(history.history['accuracy']) if a >= 0.90), 200)
        results[f'nn0_{activation}_{optimizer}'] = {
            'acc': final_acc, 'epochs_90': epochs_to_90, 'history': history
        }
        print(f"nn_0 | {activation:8} | {optimizer:4} | acc={final_acc:.3f} | epoch≥90%: {epochs_to_90}")

print()

# То же на nn_1
for activation in ['sigmoid', 'tanh']:
    for optimizer in ['adam', 'sgd']:
        model = keras.Sequential([layers.Dense(1, activation=activation, input_shape=(2,))])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(X1, y1, epochs=200, verbose=0)
        final_acc = history.history['accuracy'][-1]
        results[f'nn1_{activation}_{optimizer}'] = {
            'acc': final_acc, 'history': history
        }
        print(f"nn_1 | {activation:8} | {optimizer:4} | acc={final_acc:.3f}")

# График loss для лучших вариантов
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, prefix, title in zip(axes, ['nn0', 'nn1'], ['nn_0 (линейно разделимый)', 'nn_1 (перемешанный)']):
    for key, val in results.items():
        if key.startswith(prefix):
            label = key.replace(prefix + '_', '')
            ax.plot(val['history'].history['loss'], label=label)
    ax.set_title(title);
    ax.set_xlabel('Epoch');
    ax.set_ylabel('Loss');
    ax.legend()

plt.tight_layout()
plt.savefig('single_neuron_results.png', dpi=120)
plt.show()

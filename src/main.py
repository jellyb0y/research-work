#!./venv/bin/python

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from tensorflow.keras import Sequential, layers
from tensorflow.data import Dataset
import matplotlib.pyplot as plt

# %% [markdown]
# #### Определяем функцию, которую будем прогнозировать
# y = sin(x) * exp(-x)

# %%
func = lambda x: np.sin(x) * np.cos(x * 0.5) * np.exp(-x * 0.05)


# %%
length = 1000
x_length = 10

x_arr = np.arange(length) / x_length
y_arr = func(x_arr)

# %% [markdown]
# #### Создаём датасет для модели

# %%
train_length = int(0.7 * length)
test_length = length - train_length

input_interval = 20
output_offset = 10

datasets_train = []
labels_train = []
for i in range(train_length - output_offset - input_interval):
    data = []
    for k in range(i, i + input_interval):
        data.append(y_arr[k])
    datasets_train.append(data)
    labels_train.append([y_arr[k + output_offset]])
datasets_train = np.array(datasets_train)
labels_train = np.array(labels_train)

datasets_test = []
labels_test = []
for i in range(train_length, train_length + test_length - output_offset - input_interval):
    data = []
    for k in range(i, i + input_interval):
        data.append(y_arr[k])
    datasets_test.append(data)
    labels_test.append([y_arr[k + output_offset]])
datasets_test = np.array(datasets_test)
labels_test = np.array(labels_test)


# %%
BATCH_SIZE = 10
BUFFER_SIZE = 1000

train_univariate = Dataset.from_tensor_slices((datasets_train, labels_train))
train_univariate = train_univariate.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = Dataset.from_tensor_slices((datasets_test, labels_test))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()


# %%
model = Sequential()

model.add(layers.Dense(input_interval, input_shape=(input_interval,)))
model.add(layers.Dense(input_interval))
model.add(layers.Dense(int(input_interval / 2)))
model.add(layers.Dense(1))

model.compile(optimizer='SGD', loss='mean_squared_error')


# %%
model.fit(
    train_univariate,
    epochs=100,
    steps_per_epoch=10,
    validation_data=val_univariate,
    validation_steps=10
)


# %%
train_predictions = np.array([value[0] for value in model.predict(datasets_train)])
train_labels_ = np.array([value[0] for value in labels_train])
train_x_arr = x_arr[input_interval + output_offset:train_length]

test_predictions = np.array([value[0] for value in model.predict(datasets_test)])
test_labels_ = np.array([value[0] for value in labels_test])
test_x_arr = x_arr[train_length + input_interval + output_offset:]


# %%
plt.plot(train_x_arr, train_labels_)
plt.plot(train_x_arr, train_predictions)

plt.plot(test_x_arr, test_labels_)
plt.plot(test_x_arr, test_predictions)

plt.show()

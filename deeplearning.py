
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import data
#n = 41000*500 --- 20,500,000
#Pandas lee el csv

data = pd.read_csv('/Users/fernandasramirezm/Desktop/iA/01_data/data_stocks.csv')
#data = pd.read_csv('/Users/fernandasramirezm/Desktop/iA/01_data/pruebais.csv')

# Elimina la columna con el encabezado DATE
data = data.drop(['DATE'], 1)

# Ajusta las dimensiones del dataset
n = data.shape[0]
p = data.shape[1]

#Convierte pandas a arreglo numpy
data = data.values

# Entrenamiento
train_start = 0
#Se agarran el 80% de los datos 
train_end = int(np.floor(0.8*n))
#Inicia en lo que sobra de los datos
test_start = train_end + 1 
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

#Normaliza los datos para que queden entre -1 y 1 para una función de activación sigmoide
scaler = MinMaxScaler(feature_range=(-1, 1)) #le damos el rango para escalar
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

# Construyte x y y 
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

#Numero de stocks en el entrenamiento de los datos
n_stocks = X_train.shape[1]


n_neurons_1 = 1024*2 #2056
n_neurons_2 = 512*2
n_neurons_3 = 256*2
n_neurons_4 = 128*2

# Inicias la sesión de Tensor
net = tf.InteractiveSession()

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Initializers
#Las redes neuronales utilizan técnicas de optimización numérica, por lo que el punto de partida del 
#problema de optimización es uno de los factores clave para encontrar buenas soluciones. Hay diferentes #inicializadores disponibles en TensorFlow, cada uno con diferentes enfoques de inicialización. 
#tf.variance_scaling_initializer() es uno de los más comunes

sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Hidden weights
### Se define la arquitectura de la red o grafo. W_hidden es la b y bias_hidden es la c (y=x*b+c).
### La primera capa recibe n_stocks y saca n_neurons_1, la segunda recibe n_neurons_1 y saca n_neurons_2...
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output weights
### Defines la capa de salida de la red. 
W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
bias_out = tf.Variable(bias_initializer([1]))

# Hidden layer
### Defines las funciones u operaciones que va a realizar, en este caso: y=X*W_hidden_1+bias_hidden_1 
#generas pesos y sesgos al pasar el batch
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (transpose!)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Cost function .. busca reducir el MSE
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
### Se encarga de optimizar la red neuronal (sus pesos (b) y sus sesgos (c))
### Adam es “Adaptive Moment Estimation" y es muy usado en Deep Learning -- mse y que tanto se adapta la grafica
opt = tf.train.AdamOptimizer().minimize(mse)

# iniciar
net.run(tf.global_variables_initializer())

# grafica
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test, 'C8')
line2, = ax1.plot(y_test * 0.5,'C9')
plt.show()

#n/256 --- 
batch_size = 256
mse_train = []
mse_test = []

# Run
epochs = 15
for e in range(epochs):

    # DATOS DE ENTRENAMIENTO ALEATORIO
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        if np.mod(i, 50) == 0:
            # MSE train and test
            mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
            mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])
            # Prediction
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            #plt.paus(0.1)
            plt.pause(0.001)

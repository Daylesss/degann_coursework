import tensorflow as tf
import numpy as np
import keras
from degann.networks import IModel
from degann.networks.topology import PhysicsInformedNet
import matplotlib.pyplot as plt

np.random.seed(100)

tf.get_logger().setLevel('ERROR')

epochs = 500

x_borders = (0, 3.5)
x_col_borders = (0, 3.5)
num_points = 300

func = lambda x: np.sin(5*x)
x_data = np.linspace(*x_borders, num_points)
y_data = func(x_data) + 0.1 * np.random.randn(num_points)

x_train = tf.Variable(x_data.reshape(-1, 1), dtype=tf.float32)
y_train = tf.Variable(y_data.reshape(-1, 1), dtype=tf.float32)


def phys_loss(model: PhysicsInformedNet, tape: tf.GradientTape, x, y_pred):
    y_x = tape.gradient(y_pred, x)
    y_xx = tape.gradient(y_x, x)
    return 25*y_pred + y_xx

def boundary_dev(model: PhysicsInformedNet, tape: tf.GradientTape, x, y_pred):
    x_input = tf.constant([[0.0], [3.5]], dtype=tf.float32)
    y = model(x_input, training=True)
    return y - func(np.array([[0.0], [3.5]]))

# # Создаем экземпляр класса PINN
# pinn = IModel(
#     input_size=1,
#     block_size=[16, 16, 16, 16],
#     output_size=1,
#     phys_func=phys_loss,
#     boundary_func=boundary_dev,
#     phys_k=0.01,
#     boundary_k=0.01,
#     activation_func="tanh",
#     is_debug=True,
#     net_type="PINN"
# )

# # Компилируем модель с заданным оптимизатором
# pinn.compile(optimizer="AdamW")

# pinn.train(
#     x_data=x_train,
#     y_data=y_train,
#     epochs=epochs,
# )

# # for epoch in range(epochs):
# #     logs = pinn.train_step(dataset)
# #     if epoch % 500 == 0:
# #         print(f"Эпоха {epoch}: Потери = {logs['loss'].numpy()}")


# # Предсказываем значения на обучающей выборке
# y_pred = pinn.predict(x_train)

# # Визуализируем реальные и предсказанные данные
# plt.figure(figsize=(10, 6))
# plt.scatter(x_data, y_data, label='Actual Data', color='blue')
# plt.plot(x_data, y_pred, label='Predicted Data', color='red')
# plt.plot(x_data, func(x_data), label="Real", color="orange")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.title('Actual vs Predicted Data')
# plt.show()

# print(pinn.to_dict())


from degann.search_algorithms import pattern_search


config = {
    "loss_functions": ["MeanSquaredError"],
    "optimizers": ["Adam", "AdamW"],
    "metrics": ["MaxAbsoluteDeviation", "MeanSquaredLogarithmicError"],
    "net_shapes": [
        # [],  # neural network without hidden layers
        [10],
        [8, 8],
        [16, 16, 16, 16],
        [8, 8, 8, 8]
    ],
    "activations": ["relu", "swish", "tanh"],
    "validation_split": 0,
    "rates": [1e-2],
    "epochs": [100, 200],
    "normalize": [False],
    "use_rand_net": False,
    "net_kwargs": {
        "phys_func":phys_loss,
        "boundary_func":boundary_dev,
        "phys_k":0.01,
        "boundary_k":0.01,
    },
    "net_type": "PINN",
    "debug": True,
    
}

best_nns = pattern_search(
    x_data=x_train[:250],
    y_data=y_train[:250],
    x_val=x_train[250:],
    y_val=y_train[250:],
    **config
)

pinn = best_nns[0][-1]

print(pinn.to_dict())

# # Предсказываем значения на обучающей выборке
y_pred = pinn.predict(x_train)

# Визуализируем реальные и предсказанные данные
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Actual Data', color='blue')
plt.plot(x_data, y_pred, label='Predicted Data', color='red')
plt.plot(x_data, func(x_data), label="Real", color="orange")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Actual vs Predicted Data')
plt.show()
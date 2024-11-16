from typing import List, Optional, Dict, Callable

import tensorflow as tf

# from tensorflow import keras
import keras

from degann.networks import layer_creator


class PhysicsInformedNet(keras.Model):
    def __init__(
        self,
        input_size: int = 2,
        block_size: list = None,
        output_size: int = 10,
        phys_func: Callable | None = None,
        boundary_func: Callable | None = None,
        phys_k=0.1,
        boundary_k=1.0,
        activation_func: str = "linear",
        weight=keras.initializers.RandomUniform(minval=-1, maxval=1),
        biases=keras.initializers.RandomUniform(minval=-1, maxval=1),
        layer: str | List[str] = "Dense",
        is_debug: bool = False,
        **kwargs,
    ):
        decorator_params: List[Optional[Dict]] = [None]
        if "decorator_params" in kwargs.keys():
            decorator_params = kwargs.get("decorator_params")
            kwargs.pop("decorator_params")
        else:
            decorator_params = [None]

        if (
            isinstance(decorator_params, list)
            and len(decorator_params) == 1
            and decorator_params[0] is None
            or decorator_params is None
        ):
            decorator_params = [None] * (len(block_size) + 1)

        if (
            isinstance(decorator_params, list)
            and len(decorator_params) == 1
            and decorator_params[0] is not None
        ):
            decorator_params = decorator_params * (len(block_size) + 1)

        super(PhysicsInformedNet, self).__init__(**kwargs)
        self.blocks: List[keras.layers.Layer] = []

        if not isinstance(activation_func, list):
            activation_func = [activation_func] * (len(block_size) + 1)
        if not isinstance(layer, list):
            layer = [layer] * (len(block_size) + 1)
        if len(block_size) != 0:
            self.blocks.append(
                layer_creator.create(
                    input_size,
                    block_size[0],
                    activation=activation_func[0],
                    weight=weight,
                    layer_type=layer[0],
                    bias=biases,
                    is_debug=is_debug,
                    name=f"PINN0",
                    decorator_params=decorator_params[0],
                )
            )
            for i in range(1, len(block_size)):
                self.blocks.append(
                    layer_creator.create(
                        block_size[i - 1],
                        block_size[i],
                        activation=activation_func[i],
                        weight=weight,
                        bias=biases,
                        layer_type=layer[i],
                        is_debug=is_debug,
                        name=f"PINN{i}",
                        decorator_params=decorator_params[i],
                    )
                )
            last_block_size = block_size[-1]
        else:
            last_block_size = input_size

        self.out_layer = layer_creator.create(
            last_block_size,
            output_size,
            activation=activation_func[-1],
            weight=weight,
            bias=biases,
            layer_type=layer[-1],
            is_debug=is_debug,
            name=f"OutLayerPINN",
            decorator_params=decorator_params[-1],
        )

        self.activation_funcs = activation_func
        self.layer = layer
        self.weight_initializer = weight
        self.bias_initializer = biases
        self.input_size = input_size
        self.block_size = block_size
        self.output_size = output_size
        self.trained_time = {"train_time": 0.0, "epoch_time": [], "predict_time": 0}
        self.phys_func = phys_func
        self.phys_k = phys_k
        self.boundary_func = boundary_func
        self.boundary_k = boundary_k

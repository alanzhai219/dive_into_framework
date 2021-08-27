import numpy as np
import onnx

def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:

    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor


def main() -> None:

    # Create a dummy convolutional neural network.

    # IO tensors (ValueInfoProto).
    model_input_name = "X"
    X = onnx.helper.make_tensor_value_info(model_input_name,
                                           onnx.TensorProto.FLOAT,
                                           [1, 64, 128, 128])
    model_output_name = "Y"
    model_output_channels = 64
    Y = onnx.helper.make_tensor_value_info(model_output_name,
                                           onnx.TensorProto.FLOAT,
                                           [1, model_output_channels, 128, 128])

    # Create a Conv node (NodeProto).
    # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#conv
    conv1_output_node_name = "Conv1_Y"
    # Dummy weights for conv.
    conv1_in_channels = 64
    conv1_out_channels = 64
    conv1_kernel_shape = (3, 3)
    conv1_pads = (1, 1, 1, 1)
    conv1_W = np.ones(shape=(conv1_out_channels, conv1_in_channels,
                             *conv1_kernel_shape)).astype(np.float32)
    conv1_B = np.ones(shape=(conv1_out_channels)).astype(np.float32)
    # Create the initializer tensor for the weights.
    conv1_W_initializer_tensor_name = "Conv1_W"
    conv1_W_initializer_tensor = create_initializer_tensor(
        name=conv1_W_initializer_tensor_name,
        tensor_array=conv1_W,
        data_type=onnx.TensorProto.FLOAT)
    conv1_B_initializer_tensor_name = "Conv1_B"
    conv1_B_initializer_tensor = create_initializer_tensor(
        name=conv1_B_initializer_tensor_name,
        tensor_array=conv1_B,
        data_type=onnx.TensorProto.FLOAT)

    conv1_node = onnx.helper.make_node(
        name="Conv1",  # Name is optional.
        op_type="Conv",
        # Must follow the order of input and output definitions.
        # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
        inputs=[
            model_input_name, conv1_W_initializer_tensor_name,
            conv1_B_initializer_tensor_name
        ],
        outputs=[model_output_name],
        # The following arguments are attributes.
        kernel_shape=conv1_kernel_shape,
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        pads=conv1_pads,
    )

    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=[conv1_node],
        name="Conv",
        inputs=[X],  # Graph input
        outputs=[Y],  # Graph output
        initializer=[
            conv1_W_initializer_tensor, conv1_B_initializer_tensor,
        ],
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="onnx-conv")
    model_def.opset_import[0].version = 13

    model_def = onnx.shape_inference.infer_shapes(model_def)

    onnx.checker.check_model(model_def)

    onnx.save(model_def, "conv_1_64_128_128.onnx")


if __name__ == "__main__":

    main()
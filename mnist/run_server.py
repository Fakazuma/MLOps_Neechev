from __future__ import absolute_import

import onnx
import mlflow


def run_server():
    onnx_model = onnx.load_model('/model_repository/onnx-resnet-18/1/model.onnx')
    mlflow.onnx.save_model(
        onnx_model,
        './my_model',
    )

if __name__ == "__main__":
    main()
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import tritonclient.utils as trutils
from functools import lru_cache
import numpy as np

import cv2


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton_onnx(img: np.ndarray) -> np.ndarray:
    triton_client = get_client()
    img = np.expand_dims(np.expand_dims(img, axis=0), axis=0)
    img = img.astype('float32')

    input_text = InferInput(
        name="IMAGES", shape=list(img.shape), datatype=trutils.np_to_triton_dtype(img.dtype)
    )
    input_text.set_data_from_numpy(img, binary_data=True)

    infer_response = triton_client.infer(
        "onnx-resnet-18",
        [input_text],
        outputs=[InferRequestedOutput("CLASS_PROBS", binary_data=True)],
    )
    embeddings = infer_response.as_numpy("CLASS_PROBS")[0]
    return embeddings


def main():
    test_imgs = [cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in ['./tests/2.jpg', './tests/7.jpg']]
    test_imgs = [x / 255 for x in test_imgs]
    answers = [call_triton_onnx(img).argmax() for img in test_imgs]

    assert answers == [2, 7]


if __name__ == "__main__":
    main()

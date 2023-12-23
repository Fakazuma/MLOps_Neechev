# Классификация цифр на датасете MNIST

### Установка:

```shell
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

```shell
poetry install
```

### Заполнить конфиг (mlflow server):

`mnist/configs/config.yaml`

### Запуск обучения:

```shell
python mnist/train.py

```

### Заполнить конфиг (checkpoint path):

`mnist/configs/config.yaml`

### Инференс:

```shell
python mnist/infer.py
```
  
## Triton server
OS:
 - NAME="Ubuntu"
 - VERSION="20.04.1 LTS (Focal Fossa)"

CPU:
 - Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz
 - N = 10

RAM:
 - 64gb

Описание решаемой задачи:
 - Классификация цифр на датасете MNIST

Структура model_repository: 
```
. 
├── onnx-resnet-18  
    ├── 1
    |   └── model.onnx
    └──config.pbtxt  
```

Метрики оптимизациий: `./model_inference`





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

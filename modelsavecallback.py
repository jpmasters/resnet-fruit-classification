from tensorflow import keras, train


class ModelSaveCallback(keras.callbacks.Callback):

    def __init__(self, model, path):
        super().__init__()
        self.loss = 100.0
        self.model = model
        self.path = path

    def on_test_end(self, logs=None):
        loss = logs['loss']
        if loss < self.loss:
            self.loss = loss
            cp = train.Checkpoint(model=self.model)
            cp.write(self.path)




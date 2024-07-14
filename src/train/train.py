from src.train.batch_index_generator import BatchIndexGenerator

import matplotlib as plt


class Train:
    def __init__(self, X_train, y_train, X_test, y_test, model, optimizer, loss, metric, epoches, display=True, display_time=100):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.epoches = epoches
        self.display = display
        self.display_time = display_time
        self.train_loss = []
        self.test_loss = []
        self.train_metric = []
        self. test_metric = []

    def fit(self):
        self.train_loss = []
        self.test_loss = []
        self.train_metric = []
        self.test_metric = []
        n_size = self.X_train.shape[0]
        batch_size = 32
        batches = n_size // batch_size + (n_size % batch_size > 1)
        batch_generator = BatchIndexGenerator(n_size, batch_size)
        for i in range(self.epoches):
            for batch in range(batches):
                batch_indexes = batch_generator.next()
                X_train_batch, y_train_batch = self.X_train[batch_indexes], self.y_train[batch_indexes]
                inference = self.model.train_forward(X_train_batch)
                dy_train_batch = self.loss.gradient(inference, y_train_batch)
                self.optimizer.step_batch(dy_train_batch)

            inference_train = self.model.train_forward(self.X_train)
            train_loss = self.loss.loss(inference_train, self.y_train)
            self.train_loss.append(train_loss)
            predictions_train = self.model.predict(self.X_train)
            train_metric = self.metric(predictions_train, self.y_train)
            self.train_metric.append(train_metric)
            inference_test = self.model.train_forward(self.X_test)
            test_loss = self.loss.loss(inference_test, self.y_test)
            self.test_loss.append(test_loss)
            predictions_test = self.model.predict(self.X_test)
            test_metric = self.metric(predictions_test, self.y_test)
            self.test_metric.append(test_metric)

            if i % self.display_time == 0:
                if self.metric != None:
                    print(f"Epoch: {i}; train loss: {train_loss}, test loss: {test_loss}; "
                          f"train metric: {train_metric}; test metric: {test_metric}")
                else:
                    print(f"Epoch: {i}; train loss: {train_loss}, test loss: {test_loss};")

    def plot_learning_curves(self, metric_name="Accuracy"):
        epochs = range(1, len(self.train_loss) + 1)

        plt.figure(figsize=(14, 5))

        # Plotting training and testing loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_loss, label='Training Loss')
        plt.plot(epochs, self.test_loss, label='Testing Loss')
        plt.title('Training and Testing Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plotting training and testing metric
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_metric, label=f'Training {metric_name}')
        plt.plot(epochs, self.test_metric, label=f'Testing {metric_name}')
        plt.title(f'Training and Testing {metric_name}')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name)
        plt.legend()

        plt.tight_layout()
        plt.show()


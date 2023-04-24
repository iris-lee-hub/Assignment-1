import numpy as np
import torch
class LinearClassifier(object):
    def __init__(self, num_features = 30, num_classes = 17):
        self.num_features=num_features
        self.num_classes=num_classes
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        self.w = torch.randn([self.num_features, self.num_classes], requires_grad=True)
        
    def predict(self, X, epsilon=1e-5):
        logits = torch.matmul(X, self.w) 

        q = torch.softmax(logits, dim=0)
        return q
    def grad(self, loss):
        # Get class probabilities
        optimizer = torch.optim.SGD([self.w], lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        return self.w.grad

    def loss(self, x_fit, y_true):
        logits = torch.matmul(x_fit, self.w)
        logits = logits.reshape([x_fit.shape[0], -1])
        y_true = torch.argmax(y_true.reshape([x_fit.shape[0], -1]), dim = 1)
            
        loss = torch.nn.functional.cross_entropy(logits, y_true)

        return loss

    def fit(self, X_train, y_train, n_epochs, batch_size=2, learning_rate=1e-5):
        # Iterate over epoch
        for epoch in range(n_epochs):
            n_batches = int(np.floor(X_train.shape[0] / batch_size))

            # Generate random index
            index = np.arange(X_train.shape[0])
            np.random.shuffle(index)

            # Iterate over batches
            loss_list = []
            for batch in range(n_batches):
                beg = batch*batch_size
                end = (batch+1)*batch_size if (batch+1)*batch_size < X_train.shape[0] else -1
                X_batch = torch.tensor(X_train[beg:end],dtype=torch.float)
                y_batch = torch.tensor(y_train[beg:end],dtype=torch.float)

                # Compute the loss
                # print(X_batch.shape, y_batch.shape)
                loss = self.loss(X_batch, y_batch)
                loss_list.append(loss)

                # Compute the gradient
                gradient = self.grad(loss)

                # Compute the mean gradient over all the example images
                gradient = torch.mean(gradient, axis=0, keepdims=False)

                # Update the weights
                self.w.data = self.w.data - learning_rate * gradient

        return loss_list

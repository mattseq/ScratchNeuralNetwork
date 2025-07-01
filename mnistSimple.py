import numpy as np
from sklearn.datasets import fetch_openml

class Layer: 
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        self.biases = np.zeros(output_size)

    def forward(self, inputs):
        pass

    def backward(self, output_gradient):
        pass

class DenseLayer(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
        return self.outputs

    def backward(self, output_gradient):
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.inputs.T, output_gradient)
        biases_gradient = np.sum(output_gradient, axis=0)

        # Update weights and biases (this is a placeholder for actual optimization logic)
        self.weights -= 0.01 * weights_gradient
        self.biases -= 0.01 * biases_gradient

        return input_gradient
    
class ReluActivation:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, output_gradient):
        return output_gradient * (self.inputs > 0)
    
class SigmoidActivation:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, output_gradient):
        return output_gradient * self.output * (1 - self.output)

class SoftmaxActivation:
    def forward(self, inputs):
        self.inputs = inputs
        # Shift inputs for numerical stability (subtract max)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient):
        # Each sample's gradient is computed using the Jacobian matrix of the softmax
        # For batch inputs, do it row-wise
        batch_size = self.output.shape[0]
        input_gradient = np.zeros_like(self.output)

        for i in range(batch_size):
            y = self.output[i].reshape(-1, 1)
            jacobian = np.diagflat(y) - np.dot(y, y.T)
            input_gradient[i] = np.dot(jacobian, output_gradient[i])

        return input_gradient



class NeuralNetwork:
    def __init__(self, loss_function):
        self.layers = []
        self.loss_function = loss_function

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, output_gradient):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function.forward(outputs, targets)
        loss_gradient = self.loss_function.backward(outputs, targets)
        self.backward(loss_gradient)
        return loss
    
    def add_layer(self, layer):
        self.layers.append(layer)

class MSELoss:
    def forward(self, predictions, targets):
        return np.mean((predictions - targets) ** 2)

    def backward(self, predictions, targets):
        return 2 * (predictions - targets) / targets.size

class CrossEntropyLoss:
    def forward(self, predictions, targets):
        # Avoid log(0) by clipping values
        epsilon = 1e-12
        predictions_clipped = np.clip(predictions, epsilon, 1. - epsilon)

        # Compute cross-entropy loss
        sample_losses = -np.sum(targets * np.log(predictions_clipped), axis=1)
        return np.mean(sample_losses)

    def backward(self, predictions, targets):
        # Gradient of softmax + cross-entropy (assuming softmax already applied)
        return (predictions - targets) / predictions.shape[0]


# One-hot encode labels
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]



# Example usage
if __name__ == "__main__":

    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    mnist_data = mnist.data
    mnist_target = mnist.target.astype(np.int64)
    mnist_data = mnist_data / 255.0  # Normalize the data

    print("MNIST dataset loaded with shape:", mnist_data.shape)

    train_images = mnist_data[:60000]
    train_labels = mnist_target[:60000]

    test_images = mnist_data[60000:]
    test_labels = mnist_target[60000:]

    train_labels_one_hot = one_hot_encode(train_labels, num_classes=10)
    test_labels_one_hot = one_hot_encode(test_labels, num_classes=10)

    nn = NeuralNetwork(CrossEntropyLoss())
    nn.add_layer(DenseLayer(784, 100))
    nn.add_layer(ReluActivation())
    nn.add_layer(DenseLayer(100, 10))
    nn.add_layer(SoftmaxActivation())

    # epochs = 100  # Number of training epochs

    # for epoch in range(epochs):
    #     loss = nn.train(train_images, train_labels_one_hot)
    #     if epoch % 10 == 0:
    #         print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    # print("Training complete.")


    batch_size = 64
    epochs = 100
    num_samples = train_images.shape[0]

    for epoch in range(epochs):
        # Shuffle data at the start of each epoch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_images_shuffled = train_images[indices]
        train_labels_shuffled = train_labels_one_hot[indices]

        epoch_loss = 0
        for i in range(0, num_samples, batch_size):
            batch_inputs = train_images_shuffled[i:i + batch_size]
            batch_targets = train_labels_shuffled[i:i + batch_size]

            loss = nn.train(batch_inputs, batch_targets)
            epoch_loss += loss

        epoch_loss /= (num_samples // batch_size)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")

    print("Training complete.")

    predictions = nn.forward(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels_one_hot, axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")



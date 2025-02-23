import jax
import jax.numpy as jnp
from jax.experimental import optimizers
import numpy as np
import imageio

# Backprop NEAT Gene class
class BackpropGene:
    def __init__(self, input_size, output_size):
        self.layers = [input_size, 10, output_size]  # Define the layer structure
        self.weights = [jnp.array(np.random.randn(m, n)) for m, n in zip(self.layers[:-1], self.layers[1:])]  # Initialize weights

    def forward(self, x):
        # Forward pass
        for w in self.weights:
            x = jax.nn.relu(x @ w)  # Apply ReLU activation
        return x

    def add_layer(self, position):
        # Add a new layer
        self.layers.insert(position + 1, 5)  # Insert a new layer with 5 neurons
        new_weights = []
        for i in range(len(self.layers) - 1):
            new_weights.append(jnp.array(np.random.randn(self.layers[i], self.layers[i+1])))  # Initialize weights for the new layer structure
        self.weights = new_weights

# Evaluation function
def evaluate_classification(gene, data, labels):
    predictions = gene.forward(data)  # Get model predictions
    return 1.0 / (1.0 + jnp.mean((predictions - labels)**2))  # Calculate accuracy

# Training function
def train_backprop(gene, data, labels, epochs=100):
    # Initialize the optimizer
    opt_init, opt_update, get_params = optimizers.adam(step_size=0.001)
    opt_state = opt_init(gene.weights)

    # Training loop
    for epoch in range(epochs):
        # Update weights
        params = get_params(opt_state)
        loss = jnp.mean((gene.forward(data) - labels)**2)  # Calculate loss
        grads = jax.grad(lambda p: jnp.mean((gene.forward(data, p) - labels)**2))(params)  # Compute gradients
        opt_state = opt_update(epoch, grads, opt_state)
        gene.weights = get_params(opt_state)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# Generate circular dataset
def generate_circle_dataset(num_samples=100):
    data = []
    labels = []
    radius = 1.0
    for _ in range(num_samples):
        theta = 2 * np.pi * np.random.rand()  # Random angle
        r = radius + np.random.normal(scale=0.1)  # Random radius with noise
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        label = 1 if r > radius else 0  # Label: 1 if outside the circle, 0 if inside
        data.append([x, y])
        labels.append([label])
    return jnp.array(data), jnp.array(labels)

# Main function
def main():
    # Generate dataset
    data, labels = generate_circle_dataset()

    # Initialize gene
    gene = BackpropGene(2, 1)

    # Train
    train_backprop(gene, data, labels, epochs=500)

    # Test performance
    test_data, test_labels = generate_circle_dataset(num_samples=100)
    accuracy = evaluate_classification(gene, test_data, test_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
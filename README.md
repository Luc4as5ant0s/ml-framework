# Neural Network Implementation in TypeScript

This project contains a basic implementation of a neural network in TypeScript. It includes functionality for forward propagation, numerical gradient calculation, backpropagation, and learning.

## Features

- **Forward Propagation**: Calculate the output of the neural network given an input.
- **Numerical Gradient Calculation**: Compute gradients numerically for checking the correctness of backpropagation.
- **Backpropagation**: Perform backpropagation to calculate gradients for learning.
- **Learning**: Update the weights and biases of the neural network using the calculated gradients.

## File Structure

- `neural.ts`: Contains the main implementation of the neural network.

## How to Use

1. Clone the repository:
    ```sh
    git clone https://github.com/Luc4as5ant0s/ml-framework.git
    cd ml-framework
    ```

2. Install the necessary dependencies:
    ```sh
    npm install
    ```

3. Run the TypeScript file:
    ```sh
    npx ts-node neural.ts
    ```

## Example

Here's an example of how to initialize a neural network and perform a forward pass:

```typescript
import { initNN, forward } from './neural';

const architecture = [2, 3, 1];
const nn = initNN(architecture);

const input = [0.5, 0.8];
const output = forward(nn, input);

console.log('Output:', output);

# MNIST Handwritten Digit Classification using PyTorch

## Project Overview

This project implements a feedforward neural network using **PyTorch** to classify handwritten digits from the MNIST dataset. The model takes a grayscale image of a handwritten digit (0–9) and predicts which digit it represents.

The project demonstrates the complete deep-learning workflow, including data loading, model development, training, evaluation, saving model weights, loading a trained model, and making predictions on unseen images.

## Results

| Metric | Result |
|---|---:|
| Training accuracy | 98.29% |
| Test accuracy | 97.43% |
| Number of output classes | 10 |
| Input image size | 28 × 28 pixels |
| Hardware | CPU |
| Framework | PyTorch |

## Technologies Used

- Python 3.11
- PyTorch
- Torchvision
- Matplotlib
- MNIST dataset

## Model Architecture

The model is a fully connected feedforward neural network.

| Layer | Output size | Activation |
|---|---:|---|
| Flatten | 784 | — |
| Linear | 128 | ReLU |
| Linear | 64 | ReLU |
| Linear | 10 | Raw output scores (logits) |

The input image contains 28 × 28 = 784 pixel values. These values are flattened and passed through two hidden layers. The final layer produces 10 output scores, one for each digit from 0 to 9.

The predicted digit is the class with the highest output score.

## Dataset

The project uses the **MNIST handwritten digit dataset**, which contains grayscale images of handwritten digits.

- Training images: 60,000
- Test images: 10,000
- Image dimensions: 28 × 28 pixels
- Classes: Digits 0 through 9

The images are converted to PyTorch tensors using `torchvision.transforms.ToTensor()`.

## Training Configuration

- Loss function: Cross-Entropy Loss
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Number of epochs: 5

The training loop follows these steps:

1. Clear the previous gradients.
2. Perform a forward pass.
3. Calculate the loss.
4. Compute gradients using backpropagation.
5. Update model weights using the optimizer.
6. Track training loss and accuracy.

## Project Structure

```text
AISkills/
├── mnist_classifier.py
├── mnist_model.pth
├── README.md
└── venv/
```

- `mnist_classifier.py`: Loads the dataset, defines and trains the neural network, evaluates performance, saves model weights, reloads the model, and makes a prediction.
- `mnist_model.pth`: Saved trained model weights.
- `README.md`: Project documentation.
- `venv/`: Local Python virtual environment (not required in the Git repository).

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/harishpil/MyCollageCoding.git
cd AISkills
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install torch torchvision matplotlib
```

### 4. Run the project

```bash
python mnist_classifier.py
```

The script trains the model, reports training metrics, evaluates test accuracy, saves the trained weights, and displays a sample image with its actual and predicted digit.

**Note:** The MNIST dataset is downloaded automatically the first time the script runs, provided an internet connection is available.

## Model Saving and Loading

The trained weights are saved using:

```python
torch.save(model.state_dict(), "mnist_model.pth")
```

The weights can later be loaded into a new instance of the same model architecture:

```python
loaded_model = MNISTClassifier()
loaded_model.load_state_dict(
    torch.load(
        "mnist_model.pth",
        map_location="cpu",
        weights_only=True
    )
)
loaded_model.eval()
```

This allows the model to make predictions without training again.

## Sample Prediction

The model was tested on an unseen MNIST image.

- Actual digit: 7
- Predicted digit: 7
- Prediction result: Correct

## Key Learnings

Through this project, I gained hands-on experience with:

- Building neural networks using `torch.nn`.
- Loading datasets using `torchvision`.
- Implementing forward propagation and backpropagation.
- Training a model with the Adam optimizer.
- Evaluating a model on unseen test data.
- Saving and loading model weights.
- Performing inference using a trained model.
- Visualizing predictions using Matplotlib.
- Running deep-learning workloads on a CPU without requiring an NVIDIA GPU.

## Future Improvements

- Display multiple predictions in a grid.
- Visualize incorrect predictions and analyze model errors.
- Add a confusion matrix and classification report.
- Experiment with different architectures and hyperparameters.
- Build a simple web interface for handwritten digit prediction.
- Compare the feedforward network with a Convolutional Neural Network (CNN).

## Conclusion

This project demonstrates a complete introductory deep-learning workflow using PyTorch. The neural network achieved **97.43% test accuracy** on MNIST, successfully predicting handwritten digits from unseen examples.

It serves as a foundation for more advanced work in computer vision, neural networks, model deployment, and AI engineering.

## Author

**Harish Pilla**

Learning and building projects in Machine Learning, Deep Learning, and AI Engineering.

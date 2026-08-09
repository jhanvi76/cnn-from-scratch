# cnn-from-scratch
A hands-on implementation of CNNs — building convolution, padding, stride, and max pooling from scratch in NumPy, then training a real CNN with TensorFlow/Keras on MNIST.

# CNN From Scratch & with Keras 

A hands-on notebook that builds up the core building blocks of a **Convolutional Neural Network (CNN) from scratch using pure NumPy**, then implements and trains a **real CNN using TensorFlow/Keras on the MNIST digit dataset**.

This project is meant as a learning resource — it walks through *why* CNNs work by manually coding each operation before showing how the same ideas map onto a production deep learning framework.

## What's Inside

### Part 1 — CNN Building Blocks (Manual / NumPy)
- **Convolution operation** — sliding a filter/kernel over an image and computing the dot product
- **Padding** — preserving spatial dimensions with `np.pad`
- **Stride** — controlling how the filter moves across the image
- **Max Pooling** — downsampling feature maps
- **Full manual CNN pipeline** — chaining convolution + pooling together step by step

### Part 2 — CNN with TensorFlow/Keras (MNIST)
- Loading and preprocessing the MNIST handwritten digit dataset
- Normalizing and reshaping image data for CNN input
- Building a `Sequential` model with:
  - `Conv2D` + `MaxPooling2D` layers (feature extraction)
  - `Flatten` + `Dense` layers (classification head)
- Compiling and training the model
- Evaluating test accuracy and making predictions on unseen digits

## Repository Structure
```
.
├── CNN_from_scratch_and_keras.ipynb   # Main notebook
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── .gitignore                         # Files/folders excluded from git
```

## Setup & Installation

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. (Recommended) Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook
   ```bash
   jupyter notebook CNN_from_scratch_and_keras.ipynb
   ```

## Tech Stack
- Python 3
- NumPy
- TensorFlow / Keras
- Matplotlib
- Jupyter Notebook

## Results
The Keras CNN is trained on the MNIST dataset (60,000 training / 10,000 test images) for 10 epochs and evaluated on test accuracy, with a sample prediction demonstrated at the end of the notebook.

## Author
Jhanvi Khanna

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

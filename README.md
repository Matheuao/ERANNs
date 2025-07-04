
# ERANNs

An implementation of **ERANNs** (Efficient Residual Audio Neural Networks for Audio Pattern Recognition), built using the Keras API from TensorFlow.

This architecture is based on the article ["ERANNs: Efficient residual audio neural networks for audio pattern recognition"](https://doi.org/10.1016/j.patrec.2022.07.012), published in *Pattern Recognition Letters*.

I sincerely thank and acknowledge the authors of the original work — Sergey Verbitskiy, Vladimir Berikov, and Viacheslav Vyshegorodtsev — for their valuable contribution to the field. This implementation closely follows the architectural details presented in their publication.

---

## 📦 Installation

Clone this repository and install the package locally using `pip`:

```bash
git clone https://github.com/Matheuao/ERANNs.git
cd ERANNS
pip install .
````

> **Requirements:** Python ≥ 3.6, TensorFlow ≥ 2.4.0

If you're using a platform like [Kaggle Notebooks](https://www.kaggle.com/code), you can also install directly within a cell:

```python
!git clone https://github.com/Matheuao/ERANNs.git
!pip install ./ERANNS
```

---

## 🚀 Usage

Once installed, you can import and instantiate the ERANNs model as follows:

```python
from eranns import ERANNs

# Create an ERANNs model with sm=2, W=4, and 527 output classes
model = ERANNs(sm=2, W=4, T0=128, N=527)
model.summary()
```

---

## 📚 Parameters

The `ERANNs()` function accepts the following parameters:

| Parameter | Description                                                   |
| --------- | ------------------------------------------------------------- |
| `sm`      | Stride mode that controls temporal/frequency resolution       |
| `W`       | Base width multiplier for the number of filters in each stage |
| `T0`      | Temporal input dimension (default: 128)                       |
| `N`       | Number of output classes (default: 527)                       |

---

## 📁 Project Structure

```
ERANNs/
├── eranns/
│   ├── __init__.py              # Main interface
│   ├── layers.py                # ARB block implementation
│   └── model.py                 # ERANNs architecture definition
├───├── data/
|       ├──__init__.py           # Main interface
|       ├──esc_50.y              # Train dataset pipeline
|       └──esc_50_augmented.py   # Train dataset pipeline with data augmentation(still in development)
├── setup.py                     # Package setup script
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions or encounter issues, feel free to open a [GitHub Issue](https://github.com/your-username/eranns/issues) or submit a pull request.




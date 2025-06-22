
# ERANNs
a implementation of **ERANNs** (Efficient Residual Audio Neural Networks for Aaudio Pattern Recognition). Using the Keras API from TensorFlow.

---

## 📦 Installation

Clone this repository and install the package locally using `pip`:

```bash
git clone https://github.com/Matheuao/ERANNs.git
cd eranns
pip install .
````

> **Requirements:** Python ≥ 3.6, TensorFlow ≥ 2.4.0

If you're using a platform like [Kaggle Notebooks](https://www.kaggle.com/code), you can also install directly within a cell:

```python
!git clone https://github.com/Matheuao/ERANNs.git
!pip install ./eranns
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
eranns/
├── eranns/
│   ├── __init__.py       # Main interface
│   ├── layers.py         # ARB block implementation
│   └── model.py          # ERANNs architecture definition
├── setup.py              # Package setup script
├── requirements.txt      # Dependencies
└── README.md             # This file
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions or encounter issues, feel free to open a [GitHub Issue](https://github.com/your-username/eranns/issues) or submit a pull request.




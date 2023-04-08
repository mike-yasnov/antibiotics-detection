<h2 align="center">
Qualitative and non-quantitative determination of antibiotics in powdered milk using machine learning methods</h3>
<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
![GitHub Issues](https://img.shields.io/github/issues/mike-yasnov/detecting_antibiotics.svg)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr/mike-yasnov/detecting_antibiotics.svg)
![GitHub Downloads](https://img.shields.io/github/downloads/mike-yasnov/detecting_antibiotics/total.svg)
![Stars](https://img.shields.io/github/stars/mike-yasnov/detecting_antibiotics.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

</div>

---


## 📝 Table of Contents

- [Data](#data)
- [Models](#models)
- [Results](#results)
- [Сonclusions](#conclusions)
- [Usage](#usage)
- [Authors](#authors)

---

## 📊 Data <a name = "data"></a>

Tabular data obtained using voltammetry. Data consists of 1044 coumns and 958 rows. We use it to get an approximate result of our model

## 🧨 Models <a name = "models"></a>

### Classification 
- Convolutional NN for Tabular data [PyTorch]
- CatBoostClassifier  [CatBoost]
- XGBClassifier [XGBoost]

### Regression
- Linear Regression [Scikit-Learn]
- CatBoostRegressor [CatBoost]
- XGBRegressor [Regression]
- Lama [LightAutoMl]

### CNN for tabular data architecture 
![CNN Architecture](https://github.com/mike-yasnov/antibiotics-detection/blob/main/imgs/CNN-architecture.png?raw=true)


## 🏁 Results <a name = "results"></a>
### Classifiaction
| Metrics    | CatBoost      | XGBoost    | CNN      |
| ---------- | ------------- | ---------- | -------- |
| Accuracy   | 0.8330        | 0.9092     | 0.9557   |
| Precision  | 0.8326        | 0.9095     | 0.9553   |
| Recall     | 0.8352        | 0.9102     | 0.9549   |
| F1-measure | 0.8326        | 0.9091     | 0.9547   |

 

### Regression

| Metrics    | Linear Regression      | CatBoost    | XGBoost    | LAMA       |
| ---------- | ---------------------- | ----------- | ---------- | --------   |
| R2 score   | 0.9275                 | -0.0048     | -3.4942    | 0.0655     |
| MSE        | 2.29e-13               | 3.1886e-12  | 1.1149e-05 | 2.9652e-12 |
| MAE        | 3.56e-07               | 7.5134e-07  | 0.0033     | 7.1768e-07 |


## ⭐ Сonclusions <a name = "сonclusions"></a>

- Two ML models were trained.
- The model for classification at the moment is SoTA (State of The Art) on this dataset.

- A GUI was created for the convenience of using the models.


## 🎈 Usage <a name="usage"></a>
Create anaconda environment

```
git clone https://github.com/mike-yasnov/antibiotics-detection.git
cd antibiotics-detection
conda env create -f environment.yml
conda activate milk
```

To run GUI
```
python -m app
```

To check the project you can try files from folder *test_data*
## ✍️ Authors <a name = "authors"></a>

- [@mike-yasnov](https://github.com/mike-yasnov) - Idea & Initial work

See also the list of [contributors](https://github.com/mike-yasnov/antibiotics-detection/contributors) who participated in this project.


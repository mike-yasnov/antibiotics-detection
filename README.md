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
- [Usage](#usage)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

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


## 🏁Results <a name = "results"></a>
### Classifiaction
| Metrics    | CatBoost      | XGBoost    | CNN      |
| ---------- | ------------- | ---------- | -------- |
| Accuracy   | 0.8330        | 0.9092     | 0.9557   |
| Precision  | 0.8326        | 0.9095     | 0.9553   |
| Recall     | 0.8352        | 0.9102     | 0.9549   |
| F1-measure | 0.8326        | 0.9091     | 0.9547   |

 

### Regression

## 🎈 Usage <a name="usage"></a>

Add notes about how to use the system.


## ✍️ Authors <a name = "authors"></a>

- [@mike-yasnov](https://github.com/mike-yasnov) - Idea & Initial work

See also the list of [contributors](https://github.com/kylelobo/The-Documentation-Compendium/contributors) who participated in this project.

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- Hat tip to anyone whose code was used
- Inspiration
- References

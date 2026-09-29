# Computer Price Prediction Using Streamlit

A machine learning project that predicts the price of a laptop from its specifications. The Jupyter notebook covers data cleaning, exploratory analysis, feature engineering and a comparison of regression models, and a small Streamlit app uses the saved model to estimate the price of a configuration you pick.

## Features

- Cleans the raw laptop data (drops the index column, strips units from RAM and weight, converts types)
- Engineers new features: touchscreen and IPS flags, pixels per inch (PPI) from resolution and screen size, CPU grouped into five classes, storage split into HDD and SSD capacity, GPU brand, and OS grouped into Windows, Mac and Others/No OS/Linux
- Charts for the price distribution and for price by brand, type, touchscreen, IPS, CPU, RAM, GPU, OS and weight, plus a correlation heatmap
- Compares eight regression models on log price using R2 and MAE
- Streamlit app with dropdowns and inputs for each feature and a "Predict Price" button

## Dataset

`laptop_data.csv` has 1,303 laptops and these columns: Company, TypeName, Inches, ScreenResolution, Cpu, Ram, Memory, Gpu, OpSys, Weight and Price. There are no missing values. The single laptop with an ARM GPU is removed, which leaves 1,302 rows for modelling.

## Method

1. Parse `ScreenResolution` into touchscreen, IPS, X and Y resolution, then combine resolution and `Inches` into PPI.
2. Reduce `Cpu` to Intel Core i3 / i5 / i7, Other Intel Processor or AMD Processor.
3. Split `Memory` (for example "128GB SSD + 1TB HDD") into HDD and SSD sizes in GB. Hybrid and flash storage are dropped because of their very low correlation with price.
4. Keep only the brand from `Gpu` and group `OpSys` into three classes.
5. Use `log(Price)` as the target, since the price distribution is right-skewed.
6. Split 85/15 into train and test sets (`random_state=2`) and fit each model inside a scikit-learn `Pipeline` that one-hot encodes Company, TypeName, Cpu Brand, Gpu brand and os and passes the numeric columns through.

## Results

Correlation with price after feature engineering: RAM 0.74, SSD 0.67, PPI 0.48, IPS 0.25, weight 0.21, touchscreen 0.19 and HDD -0.10.

Test set scores from the notebook (MAE is on log price):

| Model | R2 | MAE |
|---|---|---|
| Linear Regression | 0.807 | 0.210 |
| Ridge | 0.813 | 0.209 |
| Lasso | 0.807 | 0.211 |
| KNN (k=3) | 0.802 | 0.193 |
| Decision Tree | 0.848 | 0.180 |
| SVR (RBF kernel) | 0.807 | 0.202 |
| Random Forest | 0.887 | 0.159 |
| AdaBoost | 0.793 | 0.228 |

Random Forest gave the best scores of the models in the notebook.

## Saved model

The app loads two pickle files:

- `df.pkl` is the processed dataframe. The app uses it to fill the dropdown options.
- `pipe.pkl` is the trained pipeline. It uses the same one-hot preprocessing followed by a stacking regressor (Random Forest, Gradient Boosting and XGBoost, with Ridge as the final estimator). The code that trains and saves this stacking model is not in the notebook.

The pipeline was saved with scikit-learn 1.0.2 and XGBoost 1.6.2, so `requirements.txt` pins those versions. Other versions may not be able to load the pickle.

## Tech stack

Python, pandas, NumPy, Matplotlib, seaborn, scikit-learn, XGBoost, Streamlit, Jupyter

## Project structure

```
Computer Price Prediction.ipynb   cleaning, EDA, feature engineering, model comparison
app.py                            Streamlit app
laptop_data.csv                   dataset
df.pkl                            processed dataframe used by the app
pipe.pkl                          trained model pipeline
requirements.txt                  Python dependencies
```

## Setup

Use Python 3.9 or 3.10. scikit-learn 1.0.2 only has prebuilt wheels up to Python 3.10.

```bash
git clone https://github.com/Hamza-Tahirr/Computer-Price-Prediction-Using-Streamlit.git
cd Computer-Price-Prediction-Using-Streamlit
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

No API keys or environment variables are needed.

## Usage

Start the app from the project folder, because the pickle files are loaded with relative paths:

```bash
streamlit run app.py
```

Streamlit opens the app in your browser, usually at http://localhost:8501. Choose the brand, type, RAM, weight, touchscreen, IPS, screen size, resolution, CPU, HDD, SSD, GPU and OS, then click **Predict Price**. The app works out PPI from the resolution and screen size, runs the pipeline, and converts the log prediction back to a price.

To open the notebook:

```bash
jupyter notebook "Computer Price Prediction.ipynb"
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

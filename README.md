# Car Price Prediction with Machine Learning

CodeAlpha Data Science Internship | Task 3

## Problem
Predict the selling price of a used car from features such as present (showroom) price, age, kilometres driven, fuel type, seller type, transmission and number of previous owners.

## Dataset
`car_data.csv` has 301 cars (prices are in lakhs).

## Approach
1. **Data cleaning:** checked missing values (none) and removed 2 duplicate rows.
2. **Feature engineering:** created `Car_Age`, `Kms_per_year` and `Log_Present_Price`. Dropped `Car_Name` because it has too many rare model names.
3. **EDA:** price distribution, correlation heatmap, price by fuel type, present vs selling price.
4. **Preprocessing:** one-hot encoding for categorical columns, scaling for numeric columns, log transform on the target (selling price).
5. **Models compared:** Linear Regression, Ridge, Random Forest, Gradient Boosting (5-fold cross validation).
6. **Tuning:** GridSearchCV on the Ridge `alpha`.

## Results
| Model | Test R2 | MAE (lakhs) | RMSE (lakhs) |
|---|---|---|---|
| Ridge (final) | 0.91 | 0.90 | 1.55 |
| Linear | 0.91 | 0.90 | 1.56 |
| Gradient Boosting | 0.82 | 1.03 | 2.17 |
| Random Forest | 0.62 | 1.35 | 3.12 |

## Key Insights
- Present (showroom) price is the strongest predictor of the selling price, followed by car age.
- Tree models did well in cross validation but worse on the test set, because the dataset has very few expensive cars and trees cannot extrapolate to them. The simpler Ridge model generalised best.

## How to Run
1. Install the libraries: `pip install pandas numpy scikit-learn matplotlib seaborn`
2. Open `CodeAlpha_CarPricePrediction.ipynb` in Jupyter or VS Code and click Run All.
3. Keep `car_data.csv` in the same folder.

## Tools
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

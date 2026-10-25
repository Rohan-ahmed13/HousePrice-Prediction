# House Price Prediction

A reproducible machine-learning project for estimating house prices from
[`Housing.csv`](./Housing.csv). The dataset contains 545 homes, with `price` as
the target and 12 numerical and categorical property features.

## Setup

Use Python 3.10 or newer. From this folder, create and activate a virtual
environment, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Train and evaluate

```bash
python house_price.py train
```

The command holds out 20% of the data for evaluation, reports mean absolute
error (MAE), root mean squared error (RMSE), and R², then saves a final pipeline
trained on all rows to `model/house_price_pipeline.joblib`.

To use another dataset or model location:

```bash
python house_price.py train --data Housing.csv --model model/custom.joblib --test-size 0.25
```

## Predict prices for new homes

Create a CSV with these columns (the `price` target is not needed):

```text
area,bedrooms,bathrooms,stories,parking,mainroad,guestroom,basement,hotwaterheating,airconditioning,prefarea,furnishingstatus
7420,4,2,3,2,yes,no,no,no,yes,yes,furnished
```

Then run:

```bash
python house_price.py predict --model model/house_price_pipeline.joblib --input new_homes.csv --output predictions.csv
```

The output retains the input columns and adds `predicted_price`. Categorical
fields use the values present in the dataset (for example, `yes`/`no` and
`furnished`/`semi-furnished`/`unfurnished`). Unknown categories are handled by
the pipeline without breaking predictions.

## Notebook

Open `house.ipynb` in Jupyter or VS Code for dataset exploration, visualizations,
holdout evaluation, and a sample prediction using the same reusable pipeline.

## Modeling notes

The workflow uses a fixed random seed for a reproducible 80/20 split. Numeric
features are standardized, categorical features are one-hot encoded, and a
random forest regressor is used. Preprocessing is kept inside the model
pipeline so it is fit only on training data during evaluation and is saved
together with the final estimator.

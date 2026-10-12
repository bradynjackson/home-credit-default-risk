# home-credit-default-risk

**Author:** Bradyn Jackson  
**Course:** IS 6812

## Project Description

Home Credit is a lender serving clients with little or no traditional credit
history. This data will help us see which clients are capable of repayment that
are getting rejected. It will also help us see when a model approves someone
that can't repay.

## Business Problem

Credit losses, not volume or margin, dominate Home Credit's profitability. In
the first quarter of 2019 the group recorded EUR 430 million in impairment
losses against EUR 889 million of net interest income and EUR 126 million of net
profit. Losses consumed roughly 48% of net interest income and exceeded
quarterly net profit by a factor of 3.4. A small improvement in risk assessment
moves earnings further than almost any other lever available to the business.

The exposure runs in both directions. An applicant wrongly approved becomes a
write-off. An applicant wrongly declined is foregone interest income and, for a
lender whose stated purpose is financial inclusion, a customer pushed back
toward the predatory lenders Home Credit exists to displace. Current
underwriting cannot separate these two populations as sharply as the available
data should allow.

## Project Objective

Build a model that estimates each applicant's probability of default from data
Home Credit already collects, and translate that model's performance into
projected business impact.

The output is a probability rather than an approve-or-decline verdict. That
distinction carries commercial weight: a probability lets the business set its
own approval threshold and move it as risk appetite changes, which makes the
tradeoff between losses and approvals a decision management adjusts rather than
a fixed property of the model.

Success is measured by cost of risk, meaning impairment losses as a share of the
average net loan portfolio. Model discrimination (AUC-ROC) is tracked as the
means of reaching that, not as the goal itself. Accuracy is not a useful measure
here: only 8.07% of training applicants default, so a model that approves
everyone is already 91.9% accurate and commercially worthless.

## Data

Source: [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
(Kaggle, 2018).

| File | Rows | Columns |
| --- | --- | --- |
| `application_train.csv` | 307,511 | 122 |
| `application_test.csv` | 48,744 | 121 |

`TARGET` is binary, where 1 marks a client with payment difficulties. 8.07% of
training rows are positive.

The competition also supplies six supplementary tables covering credit bureau
records, previous Home Credit applications, and instalment, credit card and POS
payment histories. These are out of scope for the current phase, which works
from the application tables alone.

**The data is not stored in this repository.** `data/` and `*.csv` are
gitignored because the files exceed GitHub's size limits. To reproduce this
work, join the Kaggle competition, download the archive, and extract
`application_train.csv` and `application_test.csv` into a `data/` folder at the
project root.

## Approach

### 1. Exploratory data analysis

`eda_home_credit.qmd` (rendered to `eda_home_credit.html`) examines the target
distribution, missingness structure, and the relationship between candidate
predictors and default. The findings that drive everything downstream:

- The three `EXT_SOURCE` external scores are the strongest individual
  predictors, correlating with `TARGET` at -0.179, -0.160 and -0.155, but they
  are missing for 56.4%, 0.2% and 19.8% of applicants.
- 41 of 122 columns are more than half empty, and the emptiest are all building
  attributes whose missingness is structural rather than random.
- `DAYS_EMPLOYED` carries a sentinel value of 365243 for 18.0% of rows, all of
  them pensioners and unemployed applicants.
- Default rates fall monotonically with age, from 12.29% at ages 20 to 25 down
  to 3.66% at 65 to 70.

### 2. Data preparation

`data_preparation.R` turns those findings into reusable functions. It is
organized to mirror the transformation itself:

| Function | Purpose |
| --- | --- |
| `read_application()` | Loads a CSV with column types inferred from a wide sample |
| `recode_sentinels()` | Converts `365243` and `"XNA"` placeholders to `NA`, flagging the sentinel rows first |
| `add_missing_indicators()` | Adds binary flags for columns whose missingness is informative |
| `engineer_features()` | Builds the derived features listed below |
| `fit_prep_params()` | Measures every parameter (medians, caps, level sets, modes) on the training data only |
| `apply_prep()` | Applies those stored parameters to any frame |
| `prepare_data()` | Runs the full pipeline over train and test |
| `validate_preparation()` | Asserts that the two outputs are consistent |

Engineered features: `AGE_YEARS`, `YEARS_EMPLOYED`, `YEARS_REGISTERED`,
`YEARS_ID_PUBLISHED`, `EMPLOYED_AGE_RATIO`, `CREDIT_INCOME_RATIO`,
`ANNUITY_INCOME_RATIO`, `CREDIT_TERM`, `CREDIT_GOODS_RATIO`,
`INCOME_PER_PERSON`, `EXT_SOURCE_MEAN`, `EXT_SOURCE_MIN`, `EXT_SOURCE_MAX`,
`EXT_SOURCE_COUNT`, and `AGE_BAND`.

**Preventing leakage.** The split between `fit_prep_params()` and `apply_prep()`
is the central design decision. Every quantity the pipeline needs is measured
once on the training data and stored, then applied unchanged to both frames.
Nothing the test set contains influences how either set is transformed. That is
also why extreme values are capped rather than dropped: `apply_prep()` runs on
test data, where every row must survive to receive a prediction.

Usage:

```r
source("data_preparation.R")

prepared <- prepare_data()
validate_preparation(prepared$train, prepared$test)

prepared$train        # 307,511 x 127, model-ready
prepared$test         # 48,744 x 126
prepared$params       # the stored transformation parameters
```

`validate_preparation()` confirms that both frames carry identical columns
(excluding `TARGET`), that categorical levels match, and that no missing values
remain.

### 3. Modeling

`modeling_home_credit.qmd` (rendered to
[`modeling_home_credit.html`](modeling_home_credit.html), a self-contained
report with code collapsed by default and a show/hide-all toggle) builds on
reusable functions in `modeling_functions.R`. Models were compared on mean AUC-ROC over
the same 3-fold stratified CV splits of a 20% training sample; hyperparameters
were searched on a 5,000-row sample, and the final model was fit on all
307,511 rows. Exploration-stage models included `CODE_GENDER`; the final row
does not.

| Model | CV AUC |
| --- | --- |
| Majority-class baseline | 0.500 |
| Random forest (500 trees) | 0.715 |
| Logistic regression, 8 EDA predictors | 0.727 |
| XGBoost, untuned | 0.735 |
| Logistic regression, all predictors | 0.742 |
| **XGBoost, tuned (500 trees), without `CODE_GENDER`** | **0.748** |

- **Class imbalance:** no strategy improved AUC. Down- and upsampling inflated
  predicted default probabilities from about 8% to 28–42%, and SMOTE produced
  synthetic defaults a model could tell apart from real ones, so no adjustment
  was used.
- **Final model:** tuned XGBoost, chosen for the highest CV AUC; it beat
  logistic regression in every fold. `CODE_GENDER` was excluded for fairness
  at a cost of 0.002 AUC. Age and education remain as potential proxies.
- **Benchmark:** logistic regression on all predictors (also without
  `CODE_GENDER`, CV AUC 0.740) is kept as the interpretable benchmark, since
  its coefficients can be read and explained.
- **Kaggle:** private AUC **0.75817** (public 0.75608), above the CV estimate,
  most likely because the final model trained on all rows rather than one CV
  fold's ~41,000.

## Repository Structure

```
home-credit-default-risk/
├── data/                     # gitignored; download from Kaggle
├── eda_home_credit.qmd       # exploratory analysis source
├── eda_home_credit.html      # rendered EDA report
├── data_preparation.R        # reusable cleaning and feature functions
├── modeling_home_credit.qmd  # modeling notebook source
├── modeling_home_credit.html # rendered modeling report
├── modeling_functions.R      # reusable sampling, recipe, model and tuning functions
├── .gitignore
└── README.md
```

## Author

Bradyn Jackson, MSBA candidate, David Eccles School of Business, University of
Utah. This repository is coursework for IS 6812, Fall 2026.

AI assistance was used for code generation, research, and drafting. All analytic
decisions, their justifications, and the written interpretation are my own.

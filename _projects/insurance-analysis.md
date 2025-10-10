---
layout: page
title: Quantile Regression
permalink: /projects/insurance-analysis/
---

# Quantile Regression 

by Sai Sri Maddirala, Yancy Castro *

Hey Everyone!

This is Sai Sri, Ari and Yancy! We want to be the voice and explain to you the concept of Quantile Regression. Hold on, it's not as difficult as it sounds. We will help you figure this out.
Disclaimer: We spent a lot of time understanding. It is easy. No pressure.
Before that, let's visit our dear friend Regression model. To not make this blog sound too technical, we considered a Kaggle dataset so it's easy to explain.

 This data deals with medical insurance charges. It looks at how personal details like age, BMI, gender, family size, and smoking habits, along with the region you live in, affect the cost of health insurance.
You might be wondering, why insurance out of all things? Well, because let's be real you, me, and pretty much everyone else pays for it at some point, so we figured it's something we can all painfully relate to.

For this analysis we considered BMI, age and smoking status because these had more effect on medical insurance charges. Let's first see what this regression depicts.Lets dive into Multiple Linear Regression model first.


## What is Multiple Linear Regression ?
Regression Model is one of the most widely used model where it estimates how several predictors together influence one outcome.

In our case,Our Model is :
$$Insurance = \beta_0 + \beta_1 \cdot Age + \beta_2 \cdot BMI + \beta_3 \cdot Smoking + \varepsilon$$

Where:
- $\beta_0$ = Intercept (baseline charges)
- $\beta_1$ = Age coefficient  
- $\beta_2$ = BMI coefficient 
- $\beta_3$ = Smoking coefficient 
- $\varepsilon$ = Error term

**Results:**
- R-squared: 0.7509
- Smoking increases charges by ~$23,848 on average
- Each year of age adds ~$256 to charges
- Each BMI unit adds ~$339 to charges

### Interactive 3D Visualization

<iframe src="/plots/mlr_3d_visualization.html" width="100%" height="600px" frameborder="0"></iframe>

The visualization shows:
- **Blue points & plane**: Non-smokers (lower charges)
- **Red points & plane**: Smokers (higher charges)
- Clear separation between smoking groups showing the dramatic impact of smoking on insurance costs

### 2. Quantile Regression Analysis

While MLR shows the average relationship, quantile regression reveals how the relationship varies across different charge levels:

<iframe src="/plots/quantile_regression.html" width="100%" height="500px" frameborder="0"></iframe>

**Key Insights:**
- **25th Percentile**: Minimal difference between smokers/non-smokers
- **50th Percentile**: Moderate smoking effect
- **75th Percentile**: Large smoking effect amplified

## Mathematical Foundation

### Quantile Loss Function

For quantile τ ∈ (0,1), the loss function is:

```
L_τ(u) = u(τ - I(u < 0))
```

Where:
- u = y - x'β (residual)
- I(·) is the indicator function

This asymmetric loss function penalizes over-prediction and under-prediction differently.

## Code Implementation

```python
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

# Load and prepare data
df = pd.read_csv("insurance.csv")
mlr_data = df[['age', 'bmi', 'smoker', 'charges']].copy()
mlr_data['smoker_yes'] = (mlr_data['smoker'] == 'yes').astype(int)

# Fit MLR model
X = sm.add_constant(mlr_data[['age', 'bmi', 'smoker_yes']])
y = mlr_data['charges']
mlr_model = sm.OLS(y, X).fit()

# Quantile regression for different percentiles
quantiles = [0.25, 0.5, 0.75]
for q in quantiles:
    qr_model = QuantReg(y, X).fit(q=q)
    # Analysis continues...
```

## Key Findings

1. **Smoking Effect Varies by Charge Level**
   - Low charges: Minimal smoking impact
   - High charges: Smoking amplifies costs significantly

2. **Age Impact is Consistent**
   - Linear relationship across all quantiles
   - Steady increase with age regardless of charge level

3. **BMI Shows Heterogeneous Effects**
   - Stronger impact at higher charge quantiles
   - Suggests BMI matters more for expensive cases

## Visualization Design Choices

- **Color Consistency**: Blue for non-smokers, red for smokers across all plots
- **Interactive Elements**: Hover details, 3D rotation, zoom capabilities
- **Transparency**: Planes at 30% opacity to show data points behind
- **Professional Styling**: Clean legends, proper labels, consistent fonts

## Technical Implementation

- **3D Surface Plots**: Generated meshgrid predictions for smooth planes
- **Regression Planes**: Separate surfaces for smoker/non-smoker groups
- **Export Functionality**: Standalone HTML files for web deployment
- **Responsive Design**: Mobile-friendly visualizations

## Conclusions

This analysis demonstrates the power of combining traditional MLR with quantile regression:

- **MLR** provides overall trends and average effects
- **Quantile Regression** reveals how relationships change across the outcome distribution
- **Smoking** is the strongest predictor, but its effect varies dramatically by charge level

The interactive visualizations make complex statistical relationships accessible and provide intuitive understanding of the data patterns.

## Next Steps

- Extend analysis to include region and children variables
- Implement machine learning models (Random Forest, XGBoost)
- Add time series analysis if longitudinal data becomes available
- Develop predictive calculator for new insurance quotes

---

*View the complete code on [GitHub](https://github.com/yourusername/QR_PROJ)*

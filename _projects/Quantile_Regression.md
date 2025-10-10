---
layout: clean
title: Quantile Regression Analysis
permalink: /projects/quantile-regression/
---

# Quantile Regression Analysis

by Sai Sri Maddirala,Jianing(Ari) Li, Yancy Castro 

Hey Everyone!

This is Sai Sri, Ari and Yancy! We want to be the voice and explain  you the concept of Quantile Regression. Hold on, it's not as difficult as it sounds. we will help you figure this out.

Disclaimer: We spent a lot of time understanding. It is easy. No pressure (haha)

Before that, let's visit our dear friend Regression model. To not make this blog sound too technical, we considered a Kaggle dataset so it's easy to explain.

This data deals with medical insurance charges. It looks at how personal details like age, BMI, gender, family size, and smoking habits, along with the region you live in, affect the cost of health insurance.
You might be wondering, why insurance out of all things? Well, because let's be real you, me, and pretty much everyone else pays for it at some point, so we figured it's something we can all painfully relate to.

For this analysis we considered BMI, age and smoking status because these had more effect on medical insurance charges. Let's first see what this regression depicts.Lets dive into Multiple Linear Regression model first.

---

## What is Multiple Linear Regression? 

Multiple Linear Regression is one of the most widely used statistical models. It estimates how several predictors work together to influence one outcome variable.

**Our Model:**

$$Insurance = \beta_0 + \beta_1 \cdot Age + \beta_2 \cdot BMI + \beta_3 \cdot Smoking + \varepsilon$$

Where:
- $\beta_0$ = Intercept (baseline charges)
- $\beta_1$ = Age coefficient  
- $\beta_2$ = BMI coefficient 
- $\beta_3$ = Smoking coefficient 
- $\varepsilon$ = Error term

**Assumptions of MLR (Multiple Linear Regression):**

1. **Linearity**: Each independent variable $X_i$ has a linear relationship with the dependent variable $Y$

2. **Normality**: The errors follow a Normal (Gaussian) distribution

3. **Homoskedasticity**: The errors have constant variance 

4. **Independence**: The errors (residuals) are independent of each other

### Interactive 3D Visualization

<iframe src="https://saisri27.github.io/qranalysis/plots/mlr_3d_visualization.html" width="100%" height="600px" frameborder="0" style="border: 2px solid #ddd; border-radius: 8px;"></iframe>

In our 3D Visualization, we mainly focussed on 3 Major predictors Age,BMI and smoking and analyzed their influence on insurance charges. The Blue plane represents non-smokers and the red plane represents smokers. The Difference between the planes above is hard to miss, The plane with smokers  is higher than the plane with non smokers. Age and BMI steadily increases the predicted charges but smoking completely changes it . By this 3D graph it is evident that smoking causes increase in medical costs.

Also, Say no to smoking !

With our trusty python and libraries , we found that for every additional year of age , the average medical bill increases by \$259 and each extra unit of BMI will add \$323 to the insurance. But Smoking ? Whoaa similar to what we have seen in the graph ,on an average smokers pay \$23,824 more than non-smokers.This says how much influence does smoking have on insurance charges.


Our Prediction model gave us a mean prediction i.e what is the expected cost given these factors . But what if we want to know more than average? What if we want a deeper analysis like how does this costs vary across people in the lowest 25% of insurance charges i.e people who pay the least or the highest 75% of insurance charges i.e people who pay highest?

To solve these issues, here comes (drum roll....) our **QUANTILE REGRESSION**!!!

![Quantile Regression Analysis](https://saisri27.github.io/qranalysis/images/img_1.jpeg){: style="width: 100%; max-width: 800px; height: auto; display: block; margin: 20px auto; border: 2px solid #ddd; border-radius: 8px;"}

The Above image says in a simplest way the difference between Linear Regression and Qunatile regression.

---

## Quantile Regression Analysis

Quantile regression does exactly as its name says , it is mainly used to estimate the distributional relationship of variables. Quantile regression can provide estimates for various quantiles such as 25th, 50th, 75th percentiles.

This approach allows us to understand how predictor effects vary across the entire distribution of insurance charges, not just the average!

## The Math behind it 

Generally, OLS minimizes the sum of squared residuals:

$$\hat{\theta}^{OLS} = \arg\min_{\theta} \sum_{i=1}^{n} (y_i - x_i^T \theta)^2$$

This treats positive and negative errors equally, both sides are penalized the same.

That’s why OLS gives us the mean line, where positive and negative residuals balance out to zero.

But in the case of Quantile Regression, the model is:

$$Q_Y(w | X) = X\theta_w$$

and the estimator is found by minimizing the quantile (pinball) loss:

$$\hat{\theta}_w = \arg\min_{\theta} \sum_{i=1}^{n} \rho_w(y_i - x_i^T \theta)$$

where the quantile loss function is:

$$\rho_w(u) = \begin{cases} 
w \cdot u & \text{if } u \geq 0 \\
(w - 1) \cdot u & \text{if } u < 0
\end{cases}$$

where:
- $y_i$ = Actual Observed value
- $x_i$ = Features (Independent Variables)
- $u$ = Residual (The Difference between actual and predicted value)
- $w$ = Quantile Level
- $\rho_w(u)$ = Quantile Loss Function (Also called as Pinball loss Function)


![Quantile Loss Function Visualization](https://saisri27.github.io/qranalysis/images/img_3.jpeg){: style="width: 100%; max-width: 800px; height: auto; display: block; margin: 20px auto; border: 2px solid #ddd; border-radius: 8px;"}

  The above loss function is also known as **quantile loss** or **pinball loss**.

**Key Insights:**

- When $w$ (Quantile Level) = 0.5, the line stays right in the middle with  half the points are above, half are below.

- When $w$ (Quantile Level) > 0.5, the model pays more attention to the points above the line. The quantile regression line will move upwards, so a larger portion of data lies below it.

- Similarly, when $w$ < 0.5, the regression will move downwards and so fewer data points lie below it.

using the interesting Quantile Regression mathematical approach, we can compute multiple regression models for each quantile to get in-depth analysis about the data. To be more precise, to get the 25th percentile model, we will train the model with w=0.25 and similarly for other percentiles.


Thanks to Python, its libraries and amazing in-built functions,we have computed Quantile Regression model and this is how the graph looks like 

<iframe src="https://saisri27.github.io/qranalysis/plots/quantile_3d_animation.html" width="100%" height="750px" frameborder="0" style="border: 2px solid #ddd; border-radius: 8px;"></iframe>


Feel free to move the slider and observe how quantile change shifts the planes (Hint: Focus more on the Red Smoker plane )

The Lower quantile predicts how  age, BMI and smoking affects charges for people who pay less

Higher Quantile predicts how these same factors  affect charge for people who pay more

As, we move the slider from lower quantile to higher quantile, we can observe initially there is small gap between the smoker and non smoker planes and eventually increases a lot. Smoking doesnt just ass a fixed cost,it makes the increase in cost from age and BMI even higher, especially for the people who already tend to have high medical expenses.This is one of the key insight that quantile regression can capture but not MLR(Multiple linear Regression).



## Benefits of Quantile Regression

Not to brag but, Quantile regression has a few advantages over MLR.

- The errors are treated differently for positive and negative errors depending on the quantile $w$. 

- Instead of giving just one regression line like MLR, it gives multiple lines/planes , one for each quantile ,  which shows how the effect of predictors changes at different levels. 

 - MLR assumes that the errors have constant variance (homoscedastic), while quantile regression can handle changing variance (heteroscedastic) too.

- Quantile Regression is also less sensitive to outliers because it  uses absolute loss instead of squared loss.

- Quantile Regression works even if the data is skewed or non-normal.


Quantile Regression is widely used to deal with Real world data, especially when we want to understand how predictors behave not just the average, also in the fields where extreme values or uneven data distributions matter such as in Finance where it is used to predict financial risk, including Value-at-Risk(VaR) and Conditional Value-at-Risk(CVaR) , Healthcarenwhere it is used to analyze relationships between variables spread across entire distribution offering deeper insights into treatment effectiveness at various stages of a disease, Environment stuides and many more.


## How to evaluate them?

- **Quantile Loss (Pinball Loss):**

$$L(w) = \frac{1}{n} \sum_{i=1}^{n} \rho_w(y_i - \hat{y}_i)$$

where:

$$\rho_w(u) = \begin{cases} 
w \cdot u & \text{if } u \geq 0 \\
(w - 1) \cdot u & \text{if } u < 0
\end{cases}$$

Lower values mean better performance for that quantile.

- **Pseudo R² (Koenker & Machado R²):**

    Koenker and Machado are the two statisticians who proposed this pseudo $R_{pseudo}^2$ that works for Quantile Regression, since the usual $R^2$ from OLS doesn’t apply here.
   This evaluation is similar to the $R^2$  that we did in MLR but this based on quantiles

$$R_{pseudo}^2 = 1 - \frac{\text{Quantile Loss of the fitted model}}{\text{Quantile Loss of a null model}}$$

Also, have a look at this if you want to know more about how the $R_{pseudo}^2$ works:
[Koenker & Machado (1999) - Inference for Quantile Regression](https://www.maths.usyd.edu.au/u/jchan/GLM/Koenker&Machado1999InferenceQuantileReg.pdf)

- **Coverage Probability :**

   This Basically tells how well the Quantile regression model matches the actual distribution of the data . When a Quantile model is built for a certain quantile, the model is supposed to predict a line or a plane in which approximately Quanatile*100% of the actual data points fall below the line

$$\text{Coverage}(w) = \frac{1}{n} \sum_{i=1}^{n} I(y_i \leq \hat{y}_i)$$

- I is an indicator function (which is =0 if true or false)
- $y_i$ = actual value
- $\hat{y}_i$ = predicted value from your quantile model

For a well calibrated model:$$\text{Coverage}(w) \approx w$$


## Common Pitfalls and Mistakes


-  **Ignoring measurement error in Dependent variable**:

   While quantile regression handles random noise it doesn’t properly handle systematic measurement errors , these systematic errors can happen due to human error, faulty sensor , rounding etc .This causes severe bias is coefficient estimates. This causes distortion and the model pulls all quantile lines/planes towards the median messing up the actual data pattern which leads to worng interpretation

- **Quantile Crossing**:

   In general , the quantiles follow the order where  25th percentile line stays below the 50th percentile and 75th percentile line stays above 50th percentile , but sometimes these lines cross each other and the order changes . 25th percentile line can shift above 50th percentile line . This is called as Quantile Crossing.This breaks the logic and Quantile crossing usually happens when the model is too simple, it has high noise or overlapping data and also when the data is trained independently and they have no connection with other quantiles.

- **Convergence Problems**:
   This problem mainly occurs when data makes it hard for the algorithm to find stable ,accurate solution. When I say , data makes it hard, it can be due to multiple reasons.
   
   some of them are:

   - Dataset is small and doesn't have enough variation. Estimating Extreme quantiles when there are few data points makes it harder for the algorithm to converge.  

   - Multicollinearity ,it results in unstable coefficient estimates and makes it harder for the model to find accurate solutions

   


## Disadvantages of Quantile Regression

   (Dont judge even smart models  have their flaws)

-  Requires Large sample size:

    Because Quantile Regression estimates quantiles ranging from low to very high percentiles, we need large sample size for getting a reliable output. Smaller datasets have fewer data points at the tails which increases uncertainity.

-   Computational complexity:

    Quantile regression is computationally complex because it needs to solve each quantile through repeated calculations , which makes it slower and harder to scale for large datasets.

-  Inefficiency in High-Dimensional settings:

   Quantile regression may become inefficient when dealing with very High dimensional predictor spaces. without any appropriate regularization techniques , the model will be complicated.


## References

- **Professor Cody's lectures and notes** - Course materials and guidance

- [**Unraveling the Mysteries of Quantile Regression**](https://juandelacalle.medium.com/unraveling-the-mysteries-of-quantile-regression-a-comprehensive-analysis-and-python-implementation-9850415cea2b) - Juan de la Calle (Medium)

- [**Introduction to Quantile Regression**](https://blog.dailydoseofds.com/p/introduction-to-quantile-regression) - Daily Dose of Data Science

- [**Errors in the Dependent Variable of Quantile Regression Models**](https://economics.mit.edu/sites/default/files/2022-09/Errors%20in%20the%20Dependent%20Variable%20of%20Quantile%20Regression%20Models.pdf) - MIT Economics Department

- [**Koenker & Machado (1999) - Inference for Quantile Regression**](https://www.maths.usyd.edu.au/u/jchan/GLM/Koenker&Machado1999InferenceQuantileReg.pdf) - University of Sydney

- [**Quantile Regression: Econometric Analysis**](http://www.econ.uiuc.edu/~roger/research/rq/QRJEP.pdf) - Roger Koenker, University of Illinois




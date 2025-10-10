---
layout: project
title: Quantile Regression
permalink: /projects/quantile-regression/
---

by Sai Sri Maddirala, Yancy Castro *

Hey Everyone!

This is Sai Sri, Ari and Yancy! We want to be the voice and explain to you the concept of Quantile Regression. Hold on, it's not as difficult as it sounds. we will help you figure this out.

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

### Interactive 3D Visualization

<iframe src="https://saisri27.github.io/linearregressionproject/plots/mlr_3d_visualization.html" width="100%" height="600px" frameborder="0" style="border: 2px solid #ddd; border-radius: 8px;"></iframe>

In our 3D Visualization, we mainly focussed on 3 Major predictors Age,BMI and smoking and analyzed their influence on insurance charges. The Blue plane represents non-smokers and the red plane represents smokers. The Difference between the planes above is hard to miss, The plane with smokers  is higher than the plane with non smokers. Age and BMI steadily increases the predicted charges but smoking completely changes it . By this 3D graph it is evident that smoking causes increase in medical costs.

Also, Say no to smoking !

With our trusty python and libraries , we found that for every additional year of age , the average medical bill increases by $259 and each extra unit of BMI will add $323 to the insurance. But Smoking ? Whoaa similar to what we have seen in the graph ,on an average smokers pay $23,824 more than non-smokers.This says how much influence does smoking have on insurance charges.


Our Prediction model gave us a mean prediction i.e what is the expected cost given these factors . But what if we want to know more than average? What if we want a deeper analysis like how does this costs vary across people in the lowest 25% or the highest 75%.

To solve these issues, here comes (drum roll....) our **QUANTILE REGRESSION**!!!

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

where :
y_i = 
x_i




![Quantile Regression Analysis](https://saisri27.github.io/linearregressionproject/images/img_1.jpeg)

The Above image says in a simplest way the difference between Linear Regression and Qunatile regression.
<iframe src="https://saisri27.github.io/linearregressionproject/plots/quantile_3d_animation.html" width="100%" height="500px" frameborder="0" style="border: 2px solid #ddd; border-radius: 8px;"></iframe>


---


*View the complete code on [GitHub](https://github.com/saisri27/linearregressionproject)*

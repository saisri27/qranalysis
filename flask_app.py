from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.utils
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import json

app = Flask(__name__)

# Load data once when app starts
df = pd.read_csv("insurance.csv")

def create_mlr_plot():
    """Create MLR 3D visualization"""
    mlr_data = df[['age', 'bmi', 'smoker', 'charges']].copy()
    mlr_data['smoker_yes'] = (mlr_data['smoker'] == 'yes').astype(int)
    X = mlr_data[['age', 'bmi', 'smoker_yes']]
    X = sm.add_constant(X)
    y = mlr_data['charges']
    mlr_model = sm.OLS(y, X).fit()

    fig = go.Figure()

    # Add data points
    non_smokers = df[df['smoker'] == 'no']
    fig.add_trace(go.Scatter3d(
        x=non_smokers['age'],
        y=non_smokers['bmi'],
        z=non_smokers['charges'],
        mode='markers',
        marker=dict(size=4, color='blue', opacity=0.7),
        name='Non-smokers',
        hovertemplate='Age: %{x}<br>BMI: %{y:.1f}<br>Charges: $%{z:,.0f}<extra></extra>'
    ))

    smokers = df[df['smoker'] == 'yes']
    fig.add_trace(go.Scatter3d(
        x=smokers['age'],
        y=smokers['bmi'],
        z=smokers['charges'],
        mode='markers',
        marker=dict(size=4, color='red', opacity=0.7),
        name='Smokers',
        hovertemplate='Age: %{x}<br>BMI: %{y:.1f}<br>Charges: $%{z:,.0f}<extra></extra>'
    ))

    # Add regression planes
    age_range = np.linspace(df['age'].min(), df['age'].max(), 20)
    bmi_range = np.linspace(df['bmi'].min(), df['bmi'].max(), 20)
    age_grid, bmi_grid = np.meshgrid(age_range, bmi_range)

    # Non-smoker plane
    smoker_0 = np.zeros_like(age_grid.flatten())
    X_pred_nonsmoker = np.column_stack([
        np.ones(len(age_grid.flatten())),
        age_grid.flatten(),
        bmi_grid.flatten(),
        smoker_0
    ])
    z_pred_nonsmoker = mlr_model.predict(X_pred_nonsmoker).reshape(age_grid.shape)

    fig.add_trace(go.Surface(
        x=age_grid, y=bmi_grid, z=z_pred_nonsmoker,
        colorscale='Blues', opacity=0.3,
        name='Non-smoker Plane',
        showscale=False,
        hovertemplate='Age: %{x}<br>BMI: %{y:.1f}<br>Predicted: $%{z:,.0f}<extra></extra>'
    ))

    # Smoker plane
    smoker_1 = np.ones_like(age_grid.flatten())
    X_pred_smoker = np.column_stack([
        np.ones(len(age_grid.flatten())),
        age_grid.flatten(),
        bmi_grid.flatten(),
        smoker_1
    ])
    z_pred_smoker = mlr_model.predict(X_pred_smoker).reshape(age_grid.shape)

    fig.add_trace(go.Surface(
        x=age_grid, y=bmi_grid, z=z_pred_smoker,
        colorscale='Reds', opacity=0.3,
        name='Smoker Plane',
        showscale=False,
        hovertemplate='Age: %{x}<br>BMI: %{y:.1f}<br>Predicted: $%{z:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title='MLR: Insurance Charges vs Age, BMI & Smoking',
        scene=dict(
            xaxis_title='Age',
            yaxis_title='BMI',
            zaxis_title='Charges ($)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=900, height=700,
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.5)",
            borderwidth=1
        )
    )
    
    return fig

def create_quantile_plot():
    """Create Quantile Regression visualization"""
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=df['age'],
        y=df['charges'],
        mode='markers',
        marker=dict(
            color=df['smoker'].map({'no': 'blue', 'yes': 'red'}),
            size=6,
            opacity=0.6
        ),
        name='Data Points',
        hovertemplate='Age: %{x}<br>Charges: $%{y:,.0f}<extra></extra>'
    ))
    
    # Add quantile regression lines
    for i, q in enumerate(quantiles):
        # Fit quantile regression
        X = sm.add_constant(df['age'])
        qr = QuantReg(df['charges'], X).fit(q=q)
        
        # Predict for age range
        age_range = np.linspace(df['age'].min(), df['age'].max(), 100)
        X_pred = sm.add_constant(age_range)
        y_pred = qr.predict(X_pred)
        
        fig.add_trace(go.Scatter(
            x=age_range,
            y=y_pred,
            mode='lines',
            line=dict(color=colors[i], width=3),
            name=f'{int(q*100)}th Percentile',
            hovertemplate=f'Age: %{{x}}<br>{int(q*100)}th Percentile: $%{{y:,.0f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Quantile Regression: Insurance Charges vs Age',
        xaxis_title='Age',
        yaxis_title='Charges ($)',
        width=900, height=600,
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.5)",
            borderwidth=1
        )
    )
    
    return fig

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/mlr')
def mlr_plot():
    """MLR visualization page"""
    fig = create_mlr_plot()
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('plot.html', graphJSON=graphJSON, title="Multiple Linear Regression")

@app.route('/quantile')
def quantile_plot():
    """Quantile regression page"""
    fig = create_quantile_plot()
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('plot.html', graphJSON=graphJSON, title="Quantile Regression")

@app.route('/api/stats')
def get_stats():
    """API endpoint for dataset statistics"""
    stats = {
        'total_records': len(df),
        'avg_age': df['age'].mean(),
        'avg_charges': df['charges'].mean(),
        'smoker_percentage': (df['smoker'] == 'yes').mean() * 100
    }
    return jsonify(stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

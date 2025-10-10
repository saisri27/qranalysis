import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg

# Page configuration
st.set_page_config(
    page_title="Insurance Charges Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 Insurance Charges Analysis: MLR & Quantile Regression")
st.markdown("""
This interactive dashboard analyzes medical insurance charges using Multiple Linear Regression (MLR) 
and Quantile Regression, focusing on age, BMI, and smoking status.
""")

# Load data
@st.cache_data
def load_data():
    """Load and cache the insurance dataset"""
    return pd.read_csv("insurance.csv")

# MLR Model
@st.cache_data
def create_mlr_model(df):
    """Create and cache the MLR model"""
    mlr_data = df[['age', 'bmi', 'smoker', 'charges']].copy()
    mlr_data['smoker_yes'] = (mlr_data['smoker'] == 'yes').astype(int)
    X = mlr_data[['age', 'bmi', 'smoker_yes']]
    X = sm.add_constant(X)
    y = mlr_data['charges']
    return sm.OLS(y, X).fit(), X, y

# Create 3D MLR visualization
def create_mlr_3d_plot(df, mlr_model):
    """Create 3D MLR visualization with regression planes"""
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
    
    # Create regression planes
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
        title='MLR with Regression Planes: Charges vs Age, BMI & Smoking',
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
            borderwidth=1,
            font=dict(size=12)
        )
    )
    
    return fig

# Create quantile regression visualization
def create_quantile_regression_plot(df):
    """Create quantile regression visualization"""
    # Prepare data for quantile regression
    non_smokers = df[df['smoker'] == 'no']
    smokers = df[df['smoker'] == 'yes']
    
    quantiles = [0.25, 0.5, 0.75]
    colors = ['lightblue', 'blue', 'darkblue']
    
    fig = go.Figure()
    
    # Add scatter plots
    fig.add_trace(go.Scatter(
        x=non_smokers['age'],
        y=non_smokers['charges'],
        mode='markers',
        marker=dict(color='lightblue', opacity=0.6, size=6),
        name='Non-smokers',
        hovertemplate='Age: %{x}<br>Charges: $%{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=smokers['age'],
        y=smokers['charges'],
        mode='markers',
        marker=dict(color='lightcoral', opacity=0.6, size=6),
        name='Smokers',
        hovertemplate='Age: %{x}<br>Charges: $%{y:,.0f}<extra></extra>'
    ))
    
    # Add quantile regression lines for non-smokers
    age_line = np.linspace(non_smokers['age'].min(), non_smokers['age'].max(), 100)
    
    for i, q in enumerate(quantiles):
        # Non-smokers
        qr_ns = QuantReg(non_smokers['charges'], sm.add_constant(non_smokers['age'])).fit(q=q)
        y_pred_ns = qr_ns.predict(sm.add_constant(age_line))
        
        fig.add_trace(go.Scatter(
            x=age_line,
            y=y_pred_ns,
            mode='lines',
            line=dict(color=colors[i], width=2),
            name=f'Q{int(q*100)} Non-smokers',
            hovertemplate=f'Age: %{{x}}<br>Q{int(q*100)} Predicted: $%{{y:,.0f}}<extra></extra>'
        ))
        
        # Smokers
        qr_s = QuantReg(smokers['charges'], sm.add_constant(smokers['age'])).fit(q=q)
        y_pred_s = qr_s.predict(sm.add_constant(age_line))
        
        fig.add_trace(go.Scatter(
            x=age_line,
            y=y_pred_s,
            mode='lines',
            line=dict(color=colors[i], width=2, dash='dash'),
            name=f'Q{int(q*100)} Smokers',
            hovertemplate=f'Age: %{{x}}<br>Q{int(q*100)} Predicted: $%{{y:,.0f}}<extra></extra>'
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

# Main app
def main():
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.header("📋 Dataset Overview")
    st.sidebar.write(f"**Total Records:** {len(df):,}")
    st.sidebar.write(f"**Age Range:** {df['age'].min()} - {df['age'].max()}")
    st.sidebar.write(f"**BMI Range:** {df['bmi'].min():.1f} - {df['bmi'].max():.1f}")
    st.sidebar.write(f"**Smokers:** {len(df[df['smoker']=='yes']):,} ({len(df[df['smoker']=='yes'])/len(df)*100:.1f}%)")
    st.sidebar.write(f"**Non-smokers:** {len(df[df['smoker']=='no']):,} ({len(df[df['smoker']=='no'])/len(df)*100:.1f}%)")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 MLR Analysis", "📈 Quantile Regression", "📋 Data Explorer"])
    
    with tab1:
        st.header("Multiple Linear Regression Analysis")
        
        # Create and display MLR model
        mlr_model, X, y = create_mlr_model(df)
        
        # Model summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R-squared", f"{mlr_model.rsquared:.4f}")
            st.metric("Adj. R-squared", f"{mlr_model.rsquared_adj:.4f}")
        
        with col2:
            st.write("**Coefficients:**")
            st.write(f"Intercept: ${mlr_model.params[0]:,.2f}")
            st.write(f"Age: ${mlr_model.params[1]:,.2f} per year")
            st.write(f"BMI: ${mlr_model.params[2]:,.2f} per unit")
            st.write(f"Smoking: ${mlr_model.params[3]:,.2f}")
        
        # 3D Plot
        st.plotly_chart(create_mlr_3d_plot(df, mlr_model), use_container_width=True)
    
    with tab2:
        st.header("Quantile Regression Analysis")
        st.write("Quantile regression shows how different quantiles of charges vary with age for smokers vs non-smokers.")
        
        # Quantile regression plot
        st.plotly_chart(create_quantile_regression_plot(df), use_container_width=True)
    
    with tab3:
        st.header("Data Explorer")
        
        # Show raw data
        if st.checkbox("Show raw data"):
            st.dataframe(df)
        
        # Basic statistics
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())
        
        # Distribution plots
        st.subheader("Charge Distribution by Smoking Status")
        fig_hist = px.histogram(
            df, x='charges', color='smoker',
            nbins=30, opacity=0.7,
            title="Distribution of Insurance Charges"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

if __name__ == "__main__":
    main()

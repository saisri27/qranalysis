# Insurance Charges Analysis Dashboard

An interactive web application for analyzing medical insurance charges using Multiple Linear Regression (MLR) and Quantile Regression.

## Features

- **3D MLR Visualization**: Interactive 3D scatter plot with regression planes for smokers and non-smokers
- **Quantile Regression Analysis**: Visualize different quantiles (25th, 50th, 75th) of charges vs age
- **Data Explorer**: Browse raw data and summary statistics
- **Interactive Dashboard**: Built with Streamlit for easy exploration

## Dataset

The app analyzes the `insurance.csv` dataset containing:
- **age**: Age of beneficiary
- **bmi**: Body mass index
- **smoker**: Smoking status (yes/no)
- **charges**: Medical charges billed by health insurance

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

## Deployment on Render

### Quick Deploy Button
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### Manual Deployment Steps

1. **Push to GitHub**:
   - Create a new repository on GitHub
   - Push this code to the repository

2. **Create Render Account**:
   - Go to [render.com](https://render.com)
   - Sign up/login with your GitHub account

3. **Deploy Web Service**:
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: `insurance-analysis-dashboard`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
     - **Plan**: Free tier

4. **Environment Variables** (Optional):
   - No additional environment variables needed

5. **Deploy**:
   - Click "Create Web Service"
   - Wait for deployment (usually 2-5 minutes)

## Files Structure

```
QR_PROJ/
├── app.py              # Main Streamlit application
├── insurance.csv       # Dataset
├── requirements.txt    # Python dependencies
├── start.sh           # Startup script
├── README.md          # This file
└── Quantile_Regression.ipynb  # Original analysis notebook
```

## Technologies Used

- **Streamlit**: Web app framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **Statsmodels**: Statistical modeling
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing

## Live Demo

Once deployed, your app will be available at:
`https://your-app-name.onrender.com`

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns

# -------------------------------------------------------------------
# PLOTLY (Interactive) VISUALIZATIONS FOR STREAMLIT
# -------------------------------------------------------------------

def plot_actual_vs_predicted(y_test, y_pred, model_name, r2):
    """Interactive scatter plot with perfect prediction line."""
    fig = px.scatter(
        x=y_test, y=y_pred,
        labels={'x': 'Actual Methane Yield', 'y': 'Predicted Methane Yield'},
        title=f"{model_name}: Actual vs Predicted<br><sup>R² = {r2:.4f}</sup>",
        opacity=0.7,
        trendline='ols'
    )
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_shape(
        type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val,
        line=dict(dash='dash', color='red'),
        name='Perfect Prediction'
    )
    fig.update_layout(height=500)
    return fig

def plot_residuals(y_pred, residuals):
    """Interactive residual diagnostic plot."""
    fig = px.scatter(
        x=y_pred, y=residuals,
        labels={'x': 'Predicted Values', 'y': 'Residuals'},
        title="Residual Plot",
        opacity=0.7
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(height=500)
    return fig

def plot_learning_curve(history):
    """Plot training history for ANN."""
    if history is None:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history.history['loss'], name='Training Loss',
        mode='lines'
    ))
    if 'val_loss' in history.history:
        fig.add_trace(go.Scatter(
            y=history.history['val_loss'], name='Validation Loss',
            mode='lines'
        ))
    fig.update_layout(
        title='Learning Curve',
        xaxis_title='Epoch',
        yaxis_title='Loss (MSE)',
        height=500
    )
    return fig

def plot_feature_importance(model, feature_names, model_name, top_n=15):
    """Interactive feature importance bar chart."""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    fig = px.bar(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        labels={'x': 'Importance', 'y': 'Feature'},
        title=f"{model_name}: Top {top_n} Feature Importance",
        color=importances[indices],
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=500)
    return fig

def plot_correlation_heatmap(df, feature_cols):
    """Interactive correlation matrix heatmap."""
    numeric_df = df[feature_cols].select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None
    
    corr_matrix = numeric_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        title="Feature Correlation Heatmap",
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    fig.update_layout(height=600)
    return fig

def plot_model_comparison(metrics_dict):
    """Grouped bar chart comparing all models."""
    models = list(metrics_dict.keys())
    r2_values = [metrics_dict[m]['R2'] for m in models]
    mae_values = [metrics_dict[m]['MAE'] for m in models]
    rmse_values = [metrics_dict[m]['RMSE'] for m in models]
    
    fig = go.Figure(data=[
        go.Bar(name='R²', x=models, y=r2_values, marker_color='green'),
        go.Bar(name='MAE', x=models, y=mae_values, marker_color='orange'),
        go.Bar(name='RMSE', x=models, y=rmse_values, marker_color='red')
    ])
    fig.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        xaxis_tickangle=-45,
        height=500
    )
    return fig

def plot_radar_chart(metrics_dict):
    """Radar chart for multi-metric comparison."""
    models = list(metrics_dict.keys())
    metrics = ['R2', 'MAE', 'RMSE']
    
    # Normalize metrics (higher is better)
    normalized = {}
    for metric in metrics:
        values = [metrics_dict[m][metric] for m in models]
        if metric == 'R2':
            min_v, max_v = min(values), max(values)
            normalized[metric] = [(v - min_v) / (max_v - min_v + 1e-6) for v in values]
        else:
            max_v = max(values)
            normalized[metric] = [1 - (v / (max_v + 1e-6)) for v in values]
    
    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_trace(go.Scatterpolar(
            r=[normalized[m][i] for m in metrics] + [normalized[metrics[0]][i]],
            theta=metrics + [metrics[0]],
            name=model,
            fill='toself',
            opacity=0.3
        ))
    
    fig.update_layout(
        title='Model Performance Radar Chart',
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=500
    )
    return fig

def plot_error_distribution(models_predictions, y_test):
    """Box plot of absolute errors across models."""
    errors = {}
    for name, y_pred in models_predictions.items():
        errors[name] = np.abs(y_test - y_pred)
    
    fig = go.Figure()
    for name, error in errors.items():
        fig.add_trace(go.Box(y=error, name=name, boxpoints='outliers'))
    
    fig.update_layout(
        title='Error Distribution Across Models',
        yaxis_title='Absolute Error',
        height=500
    )
    return fig
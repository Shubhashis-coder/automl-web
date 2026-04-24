import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from utils.data_loader import load_uploaded_file, detect_column_types
from utils.preprocess import clean_data
from utils.models import (
    train_test_scale, tune_model, train_manual, 
    evaluate_model, ensure_numeric
)
from utils.visualizations import (
    plot_actual_vs_predicted, plot_residuals, plot_learning_curve,
    plot_feature_importance, plot_correlation_heatmap, plot_model_comparison,
    plot_radar_chart, plot_error_distribution
)
from utils.helpers import (
    save_model_pipeline, load_model_pipeline, 
    save_model_zip, load_model_from_zip
)
from utils.config import DEFAULT_PARAMS, TUNING_LEVELS

# ===================================================================
# PAGE CONFIGURATION
# ===================================================================
st.set_page_config(
    page_title="Biochar AD AutoML - Methane Yield Prediction",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2ecc71;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 10px;
        border-bottom: 3px solid #2ecc71;
    }
    .best-model {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
    }
    .info-box {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2ecc71;
    }
</style>
""", unsafe_allow_html=True)

# ===================================================================
# SESSION STATE INITIALIZATION
# ===================================================================
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'df_raw': None,
        'df_clean': None,
        'target_col': None,
        'feature_cols': None,
        'scaler': None,
        'X_train_scaled': None,
        'X_test_scaled': None,
        'y_train': None,
        'y_test': None,
        'models': {},
        'model_metrics': {},
        'model_histories': {},
        'model_params': {},
        'selected_models': [],
        'trained': False,
        'loaded_model': None,
        'loaded_scaler': None,
        'loaded_features': None,
        'loaded_target': None,
        'batch_predictions': None,
        'current_page': "📂 1. Data Import",
        'test_size': 0.2
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ===================================================================
# SIDEBAR NAVIGATION
# ===================================================================
with st.sidebar:
    st.markdown("# 🧪 Biochar AD AutoML")
    st.caption("Methane Yield Prediction System")
    st.markdown("---")
    
    # Navigation options
    nav_options = [
        "📂 1. Data Import",
        "⚙️ 2. Preprocessing",
        "🤖 3. Model Training",
        "📊 4. Evaluation",
        "📈 5. Model Comparison",
        "🔮 6. Prediction",
        "💾 7. Save/Load Model"
    ]
    
    # Get current page index safely
    try:
        current_index = nav_options.index(st.session_state['current_page'])
    except (ValueError, KeyError):
        current_index = 0
        st.session_state['current_page'] = nav_options[0]
    
    # Navigation radio
    selected_page = st.radio(
        "📋 Navigation",
        nav_options,
        index=current_index,
        key='sidebar_navigation'
    )
    
    # Update current page
    if selected_page != st.session_state['current_page']:
        st.session_state['current_page'] = selected_page
        st.rerun()
    
    st.markdown("---")
    
    # Session info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Session", datetime.now().strftime('%H:%M'))
    with col2:
        if st.session_state.get('trained', False):
            st.metric("Models", len(st.session_state.get('models', {})))
    
    if st.session_state.get('df_raw') is not None:
        st.caption(f"📊 Data: {len(st.session_state['df_raw'])} rows")
    
    st.markdown("---")
    st.caption("© 2024 Biochar AD AutoML v2.0")

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================
def display_dataframe_preview(df, max_rows=100, key="preview"):
    """Display dataframe with preview."""
    st.dataframe(
        df.head(max_rows),
        use_container_width=True,
        height=300
    )
    st.caption(f"Showing {min(len(df), max_rows)} of {len(df)} rows | {len(df.columns)} columns")

def show_metric_cards(metrics):
    """Display metric cards in columns."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("R² Score", f"{metrics['R2']:.4f}")
    with col2:
        st.metric("MAE", f"{metrics['MAE']:.4f}")
    with col3:
        st.metric("RMSE", f"{metrics['RMSE']:.4f}")
    with col4:
        if 'MAPE' in metrics:
            st.metric("MAPE (%)", f"{metrics['MAPE']:.2f}%")
    with col5:
        if 'RMSE_Ratio' in metrics:
            st.metric("RMSE/Mean", f"{metrics['RMSE_Ratio']:.4f}")

def get_best_model():
    """Return the best model based on R² score."""
    if not st.session_state.get('model_metrics'):
        return None, None
    
    best_name = max(
        st.session_state['model_metrics'], 
        key=lambda x: st.session_state['model_metrics'][x]['R2']
    )
    return best_name, st.session_state['model_metrics'][best_name]

# ===================================================================
# TAB 1: DATA IMPORT
# ===================================================================
if st.session_state['current_page'] == "📂 1. Data Import":
    st.markdown('<p class="main-header">📂 Data Import</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your experimental data (Excel or CSV)",
            type=['xlsx', 'xls', 'csv'],
            help="Upload anaerobic digestion experimental data with biochar parameters"
        )
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <strong>Expected Data Format:</strong><br>
        • Columns: Process parameters<br>
        • Rows: Experimental runs<br>
        • Target: Methane yield<br>
        • Features: Biochar dose, pH, Temp, etc.
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            df, filename = load_uploaded_file(uploaded_file)
            
            if df is not None:
                st.session_state['df_raw'] = df
                st.success(f"✅ File loaded: **{filename}**")
                
                # Data statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Missing Values", int(df.isnull().sum().sum()))
                with col4:
                    st.metric("Duplicate Rows", int(df.duplicated().sum()))
                
                # Column type detection
                col_types = detect_column_types(df)
                
                st.subheader("📊 Data Preview")
                display_dataframe_preview(df)
                
                st.subheader("🔍 Column Types Detected")
                col_df = pd.DataFrame([
                    {"Column": col, "Type": dtype} 
                    for col, dtype in col_types.items()
                ])
                st.dataframe(col_df, use_container_width=True)
                
                # Column statistics
                st.subheader("📈 Numeric Column Statistics")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    elif st.session_state['df_raw'] is not None:
        st.info("Previously loaded data is available.")
        display_dataframe_preview(st.session_state['df_raw'])

# ===================================================================
# TAB 2: PREPROCESSING
# ===================================================================
elif st.session_state['current_page'] == "⚙️ 2. Preprocessing":
    st.markdown('<p class="main-header">⚙️ Data Preprocessing</p>', unsafe_allow_html=True)
    
    if st.session_state['df_raw'] is None:
        st.warning("⚠️ Please upload data in the Data Import tab first.")
        st.stop()
    
    df = st.session_state['df_raw']
    
    # Target and Feature Selection
    st.subheader("🎯 Target & Feature Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_col = st.selectbox(
            "Select Target Variable (Methane Yield)",
            options=df.columns.tolist(),
            help="This is the variable you want to predict"
        )
    
    with col2:
        available_features = [c for c in df.columns if c != target_col]
        default_features = [c for c in available_features if df[c].dtype in ['float64', 'int64']]
        
        feature_cols = st.multiselect(
            "Select Feature Columns",
            options=available_features,
            default=default_features[:min(len(default_features), 10)],
            help="Select columns to use as predictors"
        )
    
    st.markdown("---")
    
    # Preprocessing Options
    st.subheader("⚙️ Preprocessing Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        null_strategy = st.selectbox(
            "Missing Value Strategy",
            options=['mean', 'median', 'mode', 'drop', 'ffill', 'bfill', 'interpolate'],
            index=0,
            help="How to handle missing values"
        )
    
    with col2:
        outlier_cap = st.checkbox(
            "Cap Outliers (IQR Method)",
            value=True,
            help="Clip values outside 1.5 * IQR range"
        )
    
    with col3:
        encode_cat = st.checkbox(
            "Encode Categorical Variables",
            value=True,
            help="Convert text categories to numeric"
        )
    
    col4, col5 = st.columns(2)
    
    with col4:
        add_ad_features = st.checkbox(
            "Add AD-Specific Features",
            value=True,
            help="Create C/N ratio, VS/TS ratio, VFA/Alkalinity, etc."
        )
    
    with col5:
        add_biochar_features = st.checkbox(
            "Add Biochar-Specific Features",
            value=True,
            help="Create SA-PV interactions, pH-CEC index, etc."
        )
    
    # Test size
    test_size_pct = st.slider(
        "Test Set Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Percentage of data reserved for testing"
    )
    st.session_state['test_size'] = test_size_pct / 100
    
    st.markdown("---")
    
    # Apply Preprocessing
    if st.button("🔄 Apply Preprocessing & Feature Engineering", type="primary", use_container_width=True):
        if not feature_cols:
            st.error("❌ Please select at least one feature column.")
        elif not target_col:
            st.error("❌ Please select a target column.")
        elif target_col in feature_cols:
            st.error("❌ Target column cannot be in feature columns.")
        else:
            with st.spinner("Preprocessing data..."):
                try:
                    df_clean, feature_cols_updated = clean_data(
                        df, target_col,
                        null_strategy=null_strategy,
                        outlier_cap=outlier_cap,
                        encode_cat=encode_cat,
                        add_ad_features=add_ad_features,
                        add_biochar_features=add_biochar_features
                    )
                    
                    st.session_state['df_clean'] = df_clean
                    st.session_state['feature_cols'] = feature_cols_updated
                    st.session_state['target_col'] = target_col
                    
                    st.success(f"✅ Preprocessing complete!")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Initial Rows", len(df))
                    with col2:
                        st.metric("Cleaned Rows", len(df_clean))
                    with col3:
                        st.metric("Features Created", len(feature_cols_updated))
                    
                    st.subheader("📊 Cleaned Data Preview")
                    display_dataframe_preview(df_clean)
                    
                    # Feature list
                    with st.expander("📋 Complete Feature List"):
                        for i, feat in enumerate(feature_cols_updated, 1):
                            st.write(f"{i}. {feat}")
                    
                    # Correlation heatmap
                    if len(feature_cols_updated) > 1:
                        st.subheader("🔗 Feature Correlation Heatmap")
                        fig = plot_correlation_heatmap(df_clean, feature_cols_updated)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during preprocessing: {str(e)}")

# ===================================================================
# TAB 3: MODEL TRAINING
# ===================================================================
elif st.session_state['current_page'] == "🤖 3. Model Training":
    st.markdown('<p class="main-header">🤖 Model Training & Tuning</p>', unsafe_allow_html=True)
    
    if st.session_state['df_clean'] is None:
        st.warning("⚠️ Please preprocess data in the Preprocessing tab first.")
        st.stop()
    
    df = st.session_state['df_clean']
    target_col = st.session_state['target_col']
    feature_cols = st.session_state['feature_cols']
    
    # Model Selection
    st.subheader("🎯 Model Selection")
    
    available_models = [
        'Linear Regression', 'ElasticNet', 'Random Forest', 
        'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost',
        'KNN', 'SVR', 'Gaussian Process', 'ANN'
    ]
    
    selected_models = st.multiselect(
        "Select Models to Train",
        options=available_models,
        default=['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost'],
        help="Choose models for methane yield prediction"
    )
    
    st.session_state['selected_models'] = selected_models
    
    # Tuning Configuration
    st.subheader("⚙️ Hyperparameter Tuning Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tuning_method = st.selectbox(
            "Tuning Method",
            options=['optuna', 'random', 'grid'],
            index=0,
            help="Optuna: Bayesian optimization (recommended)"
        )
    
    with col2:
        tuning_level = st.selectbox(
            "Tuning Intensity",
            options=['quick', 'medium', 'extensive'],
            index=0,
            help="Higher intensity = more trials = better results but slower"
        )
    
    with col3:
        cv_folds = st.slider(
            "Cross-Validation Folds",
            min_value=2,
            max_value=10,
            value=5,
            help="More folds = more robust but slower"
        )
    
    tuning_config = TUNING_LEVELS[tuning_level]
    n_trials = tuning_config['n_trials']
    
    st.info(f"📊 **{tuning_level.upper()}** tuning: **{n_trials} trials** per model with **{cv_folds}-fold CV**")
    
    # Training
    st.markdown("---")
    
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        if not selected_models:
            st.error("❌ Please select at least one model.")
        else:
            # Prepare data
            X = df[feature_cols]
            y = df[target_col]
            
            with st.spinner("Splitting and scaling data..."):
                X_train_scaled, X_test_scaled, y_train, y_test, scaler = train_test_scale(
                    X, y, test_size=st.session_state['test_size']
                )
                
                st.session_state['X_train_scaled'] = X_train_scaled
                st.session_state['X_test_scaled'] = X_test_scaled
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['scaler'] = scaler
            
            # Train each model
            models = {}
            model_metrics = {}
            model_histories = {}
            model_params = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            training_log = st.empty()
            
            for idx, model_name in enumerate(selected_models):
                status_text.text(f"Training {model_name}... ({idx+1}/{len(selected_models)})")
                
                try:
                    if tuning_method == 'optuna':
                        model, history, best_params = tune_model(
                            model_name, X_train_scaled, y_train,
                            n_trials=n_trials,
                            progress_callback=None
                        )
                    else:
                        params = DEFAULT_PARAMS.get(model_name, {})
                        model, history = train_manual(model_name, X_train_scaled, y_train, params)
                        best_params = params
                    
                    # Evaluate
                    metrics = evaluate_model(model, X_test_scaled, y_test)
                    y_pred = metrics['y_pred']
                    
                    # Additional metrics
                    metrics['MAPE'] = mean_absolute_percentage_error(y_test, y_pred) * 100
                    metrics['RMSE_Ratio'] = metrics['RMSE'] / (y_test.mean() + 1e-10)
                    
                    models[model_name] = model
                    model_metrics[model_name] = metrics
                    model_histories[model_name] = history
                    model_params[model_name] = best_params
                    
                    training_log.success(f"✅ {model_name}: R² = {metrics['R2']:.4f}, RMSE = {metrics['RMSE']:.3f}")
                    
                except Exception as e:
                    training_log.warning(f"⚠️ Failed to train {model_name}: {str(e)}")
                
                progress_bar.progress(int((idx + 1) / len(selected_models) * 100))
            
            progress_bar.empty()
            status_text.empty()
            
            # Store results
            st.session_state['models'] = models
            st.session_state['model_metrics'] = model_metrics
            st.session_state['model_histories'] = model_histories
            st.session_state['model_params'] = model_params
            st.session_state['trained'] = True
            
            st.success(f"✅ Training complete! {len(models)} models trained successfully.")
            st.balloons()
            
            # Quick summary
            st.subheader("📊 Training Summary")
            
            summary_data = []
            for name, metrics in model_metrics.items():
                summary_data.append({
                    'Model': name,
                    'R²': f"{metrics['R2']:.4f}",
                    'MAE': f"{metrics['MAE']:.3f}",
                    'RMSE': f"{metrics['RMSE']:.3f}",
                    'MAPE (%)': f"{metrics['MAPE']:.2f}"
                })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df = summary_df.sort_values('R²', ascending=False)
                st.dataframe(summary_df, use_container_width=True)
                
                # Highlight best model
                best_name, best_metrics = get_best_model()
                if best_name:
                    st.markdown(f"""
                    <div class="best-model">
                        🏆 Best Model: <strong>{best_name}</strong> (R² = {best_metrics['R2']:.4f})
                    </div>
                    """, unsafe_allow_html=True)

# ===================================================================
# TAB 4: EVALUATION
# ===================================================================
elif st.session_state['current_page'] == "📊 4. Evaluation":
    st.markdown('<p class="main-header">📊 Model Evaluation</p>', unsafe_allow_html=True)
    
    if not st.session_state.get('trained', False):
        st.warning("⚠️ Please train models in the Model Training tab first.")
        st.stop()
    
    models = st.session_state['models']
    model_metrics = st.session_state['model_metrics']
    model_histories = st.session_state['model_histories']
    model_params = st.session_state['model_params']
    y_test = st.session_state['y_test']
    feature_cols = st.session_state['feature_cols']
    
    if not models:
        st.warning("⚠️ No models available for evaluation.")
        st.stop()
    
    # Model selector
    selected_model = st.selectbox(
        "Select Model for Detailed Evaluation",
        options=list(models.keys()),
        key='eval_model_select'
    )
    
    if selected_model:
        model = models[selected_model]
        metrics = model_metrics[selected_model]
        history = model_histories.get(selected_model)
        params = model_params.get(selected_model)
        y_pred = metrics['y_pred']
        
        st.markdown("---")
        
        # Metric Cards
        st.subheader(f"📊 {selected_model} - Performance Metrics")
        show_metric_cards(metrics)
        
        # Best Parameters
        if params:
            with st.expander("🔧 Best Hyperparameters"):
                st.json(params)
        
        st.markdown("---")
        
        # Plots
        st.subheader("📈 Visualization Dashboard")
        
        # Row 1: Actual vs Predicted + Residuals
        col1, col2 = st.columns(2)
        
        with col1:
            fig_avp = plot_actual_vs_predicted(y_test, y_pred, selected_model, metrics['R2'])
            st.plotly_chart(fig_avp, use_container_width=True)
        
        with col2:
            residuals = y_test - y_pred
            fig_res = plot_residuals(y_pred, residuals)
            st.plotly_chart(fig_res, use_container_width=True)
        
        # Feature Importance
        st.subheader("🔍 Feature Importance")
        fig_fi = plot_feature_importance(model, feature_cols, selected_model)
        if fig_fi:
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
        
        # Learning Curve (for ANN)
        if history is not None:
            st.subheader("📉 Learning Curve")
            fig_lc = plot_learning_curve(history)
            if fig_lc:
                st.plotly_chart(fig_lc, use_container_width=True)
        
        # Prediction vs Actual table
        with st.expander("📋 Prediction vs Actual Data"):
            comparison_df = pd.DataFrame({
                'Actual': y_test.values if hasattr(y_test, 'values') else y_test,
                'Predicted': y_pred,
                'Absolute Error': np.abs(y_test - y_pred),
                'Percentage Error (%)': np.abs((y_test - y_pred) / (y_test + 1e-10)) * 100
            })
            st.dataframe(comparison_df, use_container_width=True)
            
            # Download button
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                "📥 Download Predictions CSV",
                csv,
                f"{selected_model}_predictions.csv",
                "text/csv"
            )

# ===================================================================
# TAB 5: MODEL COMPARISON
# ===================================================================
elif st.session_state['current_page'] == "📈 5. Model Comparison":
    st.markdown('<p class="main-header">📈 Model Comparison Dashboard</p>', unsafe_allow_html=True)
    
    if not st.session_state.get('trained', False):
        st.warning("⚠️ Please train models in the Model Training tab first.")
        st.stop()
    
    model_metrics = st.session_state['model_metrics']
    y_test = st.session_state['y_test']
    
    if not model_metrics:
        st.warning("No metrics available.")
        st.stop()
    
    # Best Model Highlight
    best_name, best_metrics = get_best_model()
    if best_name:
        st.markdown(f"""
        <div class="best-model">
            🏆 Best Performing Model: <strong>{best_name}</strong><br>
            R² = {best_metrics['R2']:.4f} | MAE = {best_metrics['MAE']:.3f} | 
            RMSE = {best_metrics['RMSE']:.3f} | MAPE = {best_metrics.get('MAPE', 0):.2f}%
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metrics Table
    st.subheader("📊 Complete Metrics Table")
    
    metrics_table = []
    for name, metrics in model_metrics.items():
        metrics_table.append({
            'Model': name,
            'R²': metrics['R2'],
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MAPE (%)': metrics.get('MAPE', 0),
            'RMSE/Mean': metrics.get('RMSE_Ratio', 0)
        })
    
    metrics_df = pd.DataFrame(metrics_table)
    metrics_df = metrics_df.sort_values('R²', ascending=False)
    st.dataframe(metrics_df, use_container_width=True)
    
    st.markdown("---")
    
    # Comparison Charts
    st.subheader("📈 Visual Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = plot_model_comparison(model_metrics)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        fig_radar = plot_radar_chart(model_metrics)
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Error distribution
    st.subheader("📊 Error Distribution")
    models_predictions = {name: metrics['y_pred'] for name, metrics in model_metrics.items()}
    fig_err = plot_error_distribution(models_predictions, y_test)
    st.plotly_chart(fig_err, use_container_width=True)
    
    # Ranking
    st.subheader("🏆 Model Ranking")
    
    scores = {}
    for name, m in model_metrics.items():
        r2_score_norm = m['R2'] / max(met['R2'] for met in model_metrics.values())
        mape_val = m.get('MAPE', 100)
        mape_inv = 1 / (mape_val + 1e-6)
        mape_norm = mape_inv / max(1 / (met.get('MAPE', 100) + 1e-6) for met in model_metrics.values())
        rmse_val = m.get('RMSE_Ratio', 1)
        rmse_inv = 1 / (rmse_val + 1e-6)
        rmse_norm = rmse_inv / max(1 / (met.get('RMSE_Ratio', 1) + 1e-6) for met in model_metrics.values())
        
        scores[name] = 0.4 * r2_score_norm + 0.3 * mape_norm + 0.3 * rmse_norm
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (name, score) in enumerate(ranked, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        st.write(f"{medal} **{name}** - Score: {score:.3f}")

# ===================================================================
# TAB 6: PREDICTION
# ===================================================================
elif st.session_state['current_page'] == "🔮 6. Prediction":
    st.markdown('<p class="main-header">🔮 Methane Yield Prediction</p>', unsafe_allow_html=True)
    
    # Check for available models
    has_trained = st.session_state.get('trained', False) and len(st.session_state.get('models', {})) > 0
    has_loaded = st.session_state.get('loaded_model') is not None
    
    if not has_trained and not has_loaded:
        st.warning("⚠️ Please train or load a model first.")
        st.stop()
    
    # Prediction source
    pred_source = st.radio(
        "Prediction Source",
        options=["Use Trained Model", "Use Loaded Model"],
        horizontal=True,
        key='pred_source_radio'
    )
    
    if pred_source == "Use Loaded Model" and not has_loaded:
        st.warning("⚠️ No model loaded. Please load a model in the Save/Load tab.")
        st.stop()
    
    # Select model
    if pred_source == "Use Trained Model":
        model_options = list(st.session_state['models'].keys())
        selected_model = st.selectbox("Select Model", options=model_options, key='pred_model_select')
        model = st.session_state['models'][selected_model]
        scaler = st.session_state['scaler']
        feature_cols = st.session_state['feature_cols']
    else:
        model = st.session_state['loaded_model']
        scaler = st.session_state['loaded_scaler']
        feature_cols = st.session_state['loaded_features']
        selected_model = "Loaded Model"
    
    st.markdown("---")
    
    # Input method
    input_method = st.radio(
        "Input Method",
        options=["Manual Input", "Batch File Upload"],
        horizontal=True,
        key='input_method_radio'
    )
    
    if input_method == "Manual Input":
        st.subheader("📝 Enter Process Parameters")
        
        with st.form("prediction_form"):
            input_values = {}
            
            # Create columns for inputs
            num_cols = min(3, len(feature_cols))
            cols = st.columns(num_cols)
            
            for i, col_name in enumerate(feature_cols):
                with cols[i % num_cols]:
                    input_values[col_name] = st.number_input(
                        col_name,
                        value=0.0,
                        format="%.4f",
                        key=f"input_{col_name}"
                    )
            
            submitted = st.form_submit_button("🔮 Predict Methane Yield", type="primary")
            
            if submitted:
                try:
                    # Prepare input
                    input_df = pd.DataFrame([input_values])
                    input_df = input_df[feature_cols]
                    input_df = ensure_numeric(input_df)
                    
                    if scaler:
                        input_scaled = scaler.transform(input_df)
                    else:
                        input_scaled = input_df.values
                    
                    prediction = model.predict(input_scaled)
                    if prediction.ndim > 1:
                        prediction = prediction.ravel()
                    
                    st.success(f"🎯 **Predicted Methane Yield: {prediction[0]:.2f}**")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
    
    else:
        st.subheader("📤 Upload Batch File")
        
        batch_file = st.file_uploader(
            "Upload Excel/CSV file with feature values",
            type=['xlsx', 'xls', 'csv'],
            key='batch_file_uploader'
        )
        
        if batch_file is not None:
            try:
                if batch_file.name.endswith('.csv'):
                    batch_df = pd.read_csv(batch_file)
                else:
                    batch_df = pd.read_excel(batch_file, engine='openpyxl')
                
                st.subheader("📊 Uploaded Data Preview")
                display_dataframe_preview(batch_df, max_rows=5)
                
                # Check for required columns
                missing_cols = set(feature_cols) - set(batch_df.columns)
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    if st.button("🔮 Run Batch Prediction", type="primary", key='batch_predict_btn'):
                        with st.spinner("Predicting..."):
                            X_pred = batch_df[feature_cols].copy()
                            X_pred = ensure_numeric(X_pred)
                            X_pred = X_pred.fillna(X_pred.mean())
                            
                            if scaler:
                                X_scaled = scaler.transform(X_pred)
                            else:
                                X_scaled = X_pred.values
                            
                            predictions = model.predict(X_scaled)
                            if predictions.ndim > 1:
                                predictions = predictions.ravel()
                            
                            result_df = batch_df.copy()
                            result_df['Predicted_Methane_Yield'] = predictions
                            
                            st.session_state['batch_predictions'] = result_df
                            
                            st.success(f"✅ Predictions complete for {len(result_df)} samples")
                            
                            st.subheader("📊 Prediction Results")
                            display_dataframe_preview(result_df)
                            
                            # Download button
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Predictions CSV",
                                csv,
                                f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv",
                                key='download_batch_csv'
                            )
            
            except Exception as e:
                st.error(f"❌ Error processing batch file: {str(e)}")

# ===================================================================
# TAB 7: SAVE/LOAD MODEL
# ===================================================================
elif st.session_state['current_page'] == "💾 7. Save/Load Model":
    st.markdown('<p class="main-header">💾 Save & Load Models</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["💾 Save Model", "📥 Load Model", "📦 Export All"])
    
    # Save Model
    with tab1:
        st.subheader("💾 Save Single Model")
        
        if st.session_state.get('trained', False) and len(st.session_state.get('models', {})) > 0:
            model_to_save = st.selectbox(
                "Select Model to Save",
                options=list(st.session_state['models'].keys()),
                key='save_model_select'
            )
            
            save_name = st.text_input(
                "Model Name",
                value=model_to_save.replace(" ", "_"),
                key='save_model_name'
            )
            
            if st.button("💾 Save Model to Disk", type="primary", key='save_model_btn'):
                try:
                    model = st.session_state['models'][model_to_save]
                    scaler = st.session_state['scaler']
                    feature_cols = st.session_state['feature_cols']
                    target_col = st.session_state['target_col']
                    
                    path = save_model_pipeline(
                        model, scaler, feature_cols, target_col, save_name
                    )
                    
                    # Provide download
                    with open(path, 'rb') as f:
                        st.download_button(
                            "📥 Download Model File",
                            f,
                            file_name=f"{save_name}.pkl",
                            mime="application/octet-stream",
                            key='download_model_btn'
                        )
                    
                    st.success(f"✅ Model saved successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Save failed: {str(e)}")
        else:
            st.info("No trained models available. Train models first.")
    
    # Load Model
    with tab2:
        st.subheader("📥 Load Saved Model")
        
        uploaded_model = st.file_uploader(
            "Upload Model File (.pkl or .joblib)",
            type=['pkl', 'joblib'],
            key='load_model_uploader'
        )
        
        if uploaded_model is not None:
            try:
                # Save temporarily
                temp_path = f"temp_model_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_model.getbuffer())
                
                # Load pipeline
                pipeline = load_model_pipeline(temp_path)
                
                st.session_state['loaded_model'] = pipeline['model']
                st.session_state['loaded_scaler'] = pipeline['scaler']
                st.session_state['loaded_features'] = pipeline['feature_cols']
                st.session_state['loaded_target'] = pipeline['target_col']
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                st.success("✅ Model loaded successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Model Name", pipeline.get('model_name', 'Unknown'))
                    st.metric("Target Variable", pipeline['target_col'])
                with col2:
                    st.metric("Features", len(pipeline['feature_cols']))
                    with st.expander("Feature List"):
                        for feat in pipeline['feature_cols'][:20]:
                            st.write(f"- {feat}")
                        if len(pipeline['feature_cols']) > 20:
                            st.write(f"... and {len(pipeline['feature_cols']) - 20} more")
                
            except Exception as e:
                st.error(f"❌ Failed to load model: {str(e)}")
                if os.path.exists('temp_path'):
                    os.remove('temp_path')
    
    # Export All Models
    with tab3:
        st.subheader("📦 Export All Models as ZIP")
        
        if st.session_state.get('trained', False) and len(st.session_state.get('models', {})) > 0:
            if st.button("📦 Create ZIP Package", type="primary", key='create_zip_btn'):
                try:
                    zip_data = save_model_zip(
                        st.session_state['models'],
                        st.session_state['scaler'],
                        st.session_state['feature_cols'],
                        st.session_state['target_col']
                    )
                    
                    st.download_button(
                        "📥 Download All Models (ZIP)",
                        zip_data,
                        file_name=f"all_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        key='download_zip_btn'
                    )
                    
                    st.success(f"✅ ZIP package created with {len(st.session_state['models'])} models!")
                    
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
        else:
            st.info("No trained models available.")
        
        st.markdown("---")
        st.subheader("📥 Load Models from ZIP")
        
        uploaded_zip = st.file_uploader(
            "Upload Models ZIP Package",
            type=['zip'],
            key='load_zip_uploader'
        )
        
        if uploaded_zip is not None:
            try:
                models, scaler, feature_cols, target_col = load_model_from_zip(uploaded_zip)
                
                st.session_state['models'] = models
                st.session_state['scaler'] = scaler
                st.session_state['feature_cols'] = feature_cols
                st.session_state['target_col'] = target_col
                st.session_state['trained'] = True
                
                st.success(f"✅ Loaded {len(models)} models from ZIP!")
                
                st.write("**Loaded Models:**")
                for name in models.keys():
                    st.write(f"- {name}")
                
                st.write(f"**Target:** {target_col}")
                st.write(f"**Features:** {len(feature_cols)}")
                
            except Exception as e:
                st.error(f"❌ Failed to load ZIP: {str(e)}")
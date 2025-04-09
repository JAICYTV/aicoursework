import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import plotly.figure_factory as ff
from PIL import Image
import io
import base64

# Set page configuration
# Set page configuration with dark theme
st.set_page_config(
    page_title="Injection Molding Quality Prediction Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# Force dark theme
st.markdown("""
    <style>
        .reportview-container {
            background-color: #0e1117;
            color: #fafafa;
        }
        .sidebar .sidebar-content {
            background-color: #0e1117;
            color: #fafafa;
        }
        .Widget>label {
            color: #fafafa;
        }
        .st-bb {
            background-color: #0e1117;
        }
        .st-at {
            background-color: #0e1117;
        }
        .st-af {
            background-color: #0e1117;
        }
        .css-1aumxhk {
            color: #fafafa;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
    /* Global dark background */
    body, .main, .reportview-container, .stApp {
        background-color: #121212 !important;
        color: #E0E0E0 !important;
    }

    /* Sidebar background and text */
    .css-1d391kg, .css-1lcbmhc, .css-1cypcdb {
        background-color: #1E1E1E !important;
        color: #E0E0E0 !important;
    }

    /* Sidebar title and label */
    .css-hxt7ib, .css-qcqlej {
        color: #E0E0E0 !important;
    }

    /* General widget labels */
    label, .stSlider label, .stSelectbox label {
        color: #E0E0E0 !important;
        font-weight: 500;
    }

    /* Markdown inside Streamlit */
    .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #E0E0E0 !important;
    }

    /* Metric cards */
    .metric-card {
        background-color: #1F1F1F;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.4);
        color: #E0E0E0;
        margin-bottom: 1.5rem;
    }

    /* Plotly modebar fix */
    .js-plotly-plot .plotly .modebar {
        background-color: #2E2E2E !important;
    }

    /* Highlight text like model names */
    .highlight {
        color: #FF9800;
        font-weight: 600;
    }

    /* Headings */
    .main-header {
        font-size: 2.5rem;
        color: #90CAF9;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.6rem;
        color: #42A5F5;
        margin-bottom: 1rem;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2196F3;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }

    .stButton > button:hover {
        background-color: #1976D2;
        color: white;
    }

    /* Slider track */
    .stSlider > div > div > div {
        background: #42A5F5 !important;
    }

    /* Prediction result box */
    .prediction-box {
        background-color: #263238;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 3px 6px rgba(0,0,0,0.3);
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the preprocessed data"""
    try:
        df = pd.read_csv('preprocessed_data.csv')
        return df
    except FileNotFoundError:
        st.error("Preprocessed data file not found. Please upload the file.")
        return None

@st.cache_resource
def load_models():
    """Load the trained models"""
    try:
        with open('models.pkl', 'rb') as f:
            models = pickle.load(f)
        return models
    except FileNotFoundError:
        st.error("Models file not found. Please upload the models.pkl file or continue with limited functionality.")
        return {}
    except Exception as e:
        st.error(f"Error loading models: {str(e)}. Continuing with limited functionality.")
        return {}

@st.cache_resource
def load_preprocessing_objects():
    """Load the preprocessing objects (scaler, etc.)"""
    try:
        with open('preprocessed_split_data.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['scaler']
    except FileNotFoundError:
        st.error("Preprocessing data not found. Please upload the preprocessed_split_data.pkl file.")
        return None

@st.cache_data
def load_hypothesis_testing_results():
    """Load hypothesis testing results for ANOVA plots"""
    try:
        with open('hypothesis_testing_results.pkl', 'rb') as f:
            results = pickle.load(f)
        return results
    except FileNotFoundError:
        # If file doesn't exist, return None and we'll handle this later
        return None
    except ModuleNotFoundError as e:
        st.error(f"Missing dependency: {str(e)}. Please install the required package.")
        return None
    except Exception as e:
        st.error(f"Error loading hypothesis testing results: {str(e)}.")
        return None
def get_quality_label(quality_class):
    """Map numerical class to label"""
    quality_mapping = {
        0: "Waste",
        1: "Acceptable",
        2: "Target",
        3: "Inefficient"
    }
    return quality_mapping.get(int(quality_class), "Unknown")

def get_quality_description(quality_class):
    """Get detailed description for each quality class"""
    descriptions = {
        0: "Product fails to meet basic standards and must be scrapped.",
        1: "Product meets minimum quality standards but is not ideal.",
        2: "Product meets the desired quality specifications.",
        3: "Product is above acceptable but falls short of target quality due to process inefficiencies."
    }
    return descriptions.get(int(quality_class), "Unknown classification")

def plot_feature_importance(model_name, importance, features, top_n=15):
    """Plot feature importance for the selected model"""
    # Create a DataFrame for the feature importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False).head(top_n)
    
    # Create a horizontal bar chart
    fig = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title=f'Top {top_n} Features for {model_name}',
        color='Importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=500,
        width=800,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def get_feature_ranges(df):
    """Get min, max, and default values for each feature"""
    feature_ranges = {}
    for col in df.columns:
        if col != 'quality':
            feature_ranges[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'default': float(df[col].median()),
                'step': float((df[col].max() - df[col].min()) / 100)
            }
    return feature_ranges

def plot_confusion_matrix(cm, classes):
    """Create a plotly heatmap of the confusion matrix"""
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create annotation text
    annotations = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            annotations.append({
                'x': j,
                'y': i,
                'text': f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})",
                'font': {'color': 'white' if cm_norm[i, j] > 0.5 else 'black'},
                'showarrow': False
            })
    
    # Create the heatmap
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=classes,
        y=classes,
        annotation_text=[[f"{val}" for val in row] for row in cm],
        colorscale='Blues'
    )
    
    # Update the layout
    fig.update_layout(
        title='Confusion Matrix',
        xaxis=dict(title='Predicted label'),
        yaxis=dict(title='True label', autorange='reversed'),
        width=600,
        height=600,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def plot_anova_results(df, feature, quality_column='quality'):
    """Create a box plot for ANOVA results"""
    # Create a Figure
    fig = go.Figure()
    
    # Define quality class names
    quality_names = ['Waste', 'Acceptable', 'Target', 'Inefficient']
    
    # Add box plot for each quality class
    for i, quality in enumerate(sorted(df[quality_column].unique())):
        subset = df[df[quality_column] == quality][feature]
        fig.add_trace(go.Box(
            y=subset, 
            name=quality_names[i], 
            boxpoints='outliers',
            jitter=0.3,
            marker=dict(size=3),
            line=dict(width=2)
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Distribution of {feature} by Quality Class',
        yaxis_title=feature,
        xaxis_title='Quality Class',
        boxmode='group',
        height=500,
        width=800
    )
    
    return fig

def get_class_metrics(y_true, y_pred):
    """Calculate metrics for each class"""
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    class_metrics = []
    classes = sorted(np.unique(y_true))
    
    for cls in classes:
        # Calculate metrics for this class
        true_positive = np.sum((y_true == cls) & (y_pred == cls))
        false_positive = np.sum((y_true != cls) & (y_pred == cls))
        false_negative = np.sum((y_true == cls) & (y_pred != cls))
        
        # Avoid division by zero
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics.append({
            'Class': get_quality_label(cls),
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
    
    return pd.DataFrame(class_metrics)

def calculate_process_metrics(df):
    """Calculate overall process metrics"""
    quality_counts = df['quality'].value_counts().sort_index()
    total_products = df.shape[0]
    
    # Calculate percentages
    quality_percentages = (quality_counts / total_products * 100).round(1)
    
    # Create a dictionary of metrics
    metrics = {
        'waste_rate': quality_percentages[0] if 0 in quality_percentages else 0,
        'acceptable_rate': quality_percentages[1] if 1 in quality_percentages else 0,
        'target_rate': quality_percentages[2] if 2 in quality_percentages else 0,
        'inefficient_rate': quality_percentages[3] if 3 in quality_percentages else 0,
        'total_products': total_products,
        'waste_count': quality_counts[0] if 0 in quality_counts else 0,
        'acceptable_count': quality_counts[1] if 1 in quality_counts else 0,
        'target_count': quality_counts[2] if 2 in quality_counts else 0,
        'inefficient_count': quality_counts[3] if 3 in quality_counts else 0
    }
    
    return metrics

def predict_quality(input_data, model, scaler):
    """Make prediction based on input parameters"""
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply feature engineering (same as in the preprocessing)
    # Create new feature: Pressure to Temperature Ratio
    input_df['Pressure_Temp_Ratio'] = input_df['APVs - Specific injection pressure peak value'] / input_df['Melt temperature']
    
    # Create new feature: Efficiency Ratio
    input_df['Efficiency_Ratio'] = input_df['ZUx - Cycle time'] / input_df['ZDx - Plasticizing time']
    
    # Create new feature: Power Index
    input_df['Power_Index'] = input_df['Ms - Torque peak value current cycle'] * input_df['Mm - Torque mean value current cycle']
    
    # Create new feature: Pressure Difference
    input_df['Pressure_Difference'] = input_df['APVs - Specific injection pressure peak value'] - input_df['APSs - Specific back pressure peak value']
    
    # Create new feature: Temperature Interaction
    input_df['Temp_Interaction'] = input_df['Melt temperature'] * input_df['Mold temperature']
    
    # Create new feature: Volume to Fill Time Ratio
    input_df['Volume_Fill_Ratio'] = input_df['SVo - Shot volume'] / input_df['time_to_fill']
    
    # Create polynomial features for important variables
    critical_features = ['Melt temperature', 'Mold temperature', 'APVs - Specific injection pressure peak value', 'time_to_fill']
    for feature in critical_features:
        input_df[f'{feature}_squared'] = input_df[feature] ** 2
    
    # Create a process stability indicator
    input_df['Process_Stability'] = ((input_df['SKs - Clamping force peak value'] / input_df['SKx - Closing force']) *
                                    (input_df['Ms - Torque peak value current cycle'] / input_df['Mm - Torque mean value current cycle']))
            # Add missing columns with default value
    input_df['isolation_outlier'] = 0
    input_df['lof_outlier'] = 0
        # Ensure all expected features exist and match order
    expected_features = scaler.feature_names_in_

    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0  # or a default value

    # Drop any extra columns not seen during training
    input_df = input_df[expected_features]

    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    if model.__class__.__name__ == 'Sequential':
        # Neural network model
        probabilities = model.predict(input_scaled)[0]
        prediction = np.argmax(probabilities)
        return prediction, probabilities
    else:
        # Standard sklearn model
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        return prediction, probabilities


# Example: Update your gauge chart function
def create_gauge_chart(value, title, min_val=0, max_val=100):
    """Create a gauge chart for displaying metrics"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'color': '#ffffff'}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickfont': {'color': '#ffffff'}},
            'bar': {'color': "#4287f5"},
            'steps': [
                {'range': [0, max_val/3], 'color': "#1a2639"},
                {'range': [max_val/3, 2*max_val/3], 'color': "#304669"}
            ],
            'threshold': {
                'line': {'color': "#ff5757", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        },
        number={'font': {'color': '#ffffff'}}
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.2)',
        font=dict(color='white')
    )
    
    return fig

# Load data and models
df = load_data()
models = load_models()
scaler = load_preprocessing_objects()
hypothesis_results = load_hypothesis_testing_results()

# Check if data is loaded successfully
if df is None or models is None or scaler is None:
    st.error("Failed to load required data. Please ensure all necessary files are available.")
    st.stop()

# Get feature ranges for input sliders
feature_ranges = get_feature_ranges(df)

# Get the best model based on notebook analysis (Random Forest was the best model in the notebook)
best_model = models.get('lightgbm')
if not best_model:
    # Fallback to another model if Random Forest is not available
    best_model = next(iter(models.values()))
    best_model_name = list(models.keys())[0]
else:
    best_model_name = 'Light GBM'

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Model Performance", "Statistical Analysis"])

# Main content based on selected page
if page == "Home":
    st.markdown("<h1 class='main-header'>Plastic Injection Molding Quality Prediction Dashboard</h1>", unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <p class='info-text'>
    This dashboard provides tools for predicting and analyzing quality in the plastic injection molding process.
    Use the sidebar navigation to explore different features.
    </p>
    """, unsafe_allow_html=True)
    
    # Overview of the system
    st.markdown("<h2 class='sub-header'>System Overview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
        <h3>Quality Classification</h3>
        <p>The system predicts product quality in four categories:</p>
        <ul>
            <li><b>Waste</b>: Product fails to meet basic standards</li>
            <li><b>Acceptable</b>: Meets minimum quality standards</li>
            <li><b>Target</b>: Meets desired quality specifications</li>
            <li><b>Inefficient</b>: Above acceptable but falls short of target quality</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
        <h3>Key Features</h3>
        <ul>
            <li>Real-time quality prediction based on process parameters</li>
            <li>Process monitoring and quality distribution analysis</li>
            <li>Model performance evaluation and metrics</li>
            <li>Statistical analysis of process parameters</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset summary
    st.markdown("<h2 class='sub-header'>Dataset Summary</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate process metrics
    metrics = calculate_process_metrics(df)
    
    with col1:
        st.metric("Total Samples", metrics['total_products'])
    
    with col2:
        st.metric("Features", len(df.columns) - 1)  # Exclude quality column
    
    with col3:
        st.metric("Target Quality Rate", f"{metrics['target_rate']}%")
    
    with col4:
        st.metric("Waste Rate", f"{metrics['waste_rate']}%")
    
    # Quality distribution chart
    st.subheader("Quality Distribution")
    
    quality_df = pd.DataFrame({
        'Quality Class': ['Waste', 'Acceptable', 'Target', 'Inefficient'],
        'Count': [metrics['waste_count'], metrics['acceptable_count'], 
                  metrics['target_count'], metrics['inefficient_count']]
    })
    
    fig = px.bar(
        quality_df, 
        x='Quality Class', 
        y='Count',
        color='Quality Class',
        color_discrete_sequence=px.colors.qualitative.Set1,
        text='Count'
    )
    
    fig.update_layout(
        xaxis_title='Quality Class',
        yaxis_title='Count',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Best model information
    st.markdown("<h2 class='sub-header'>Model Information</h2>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='metric-card'>
    <p>The dashboard uses a <span class='highlight'>{best_model_name}</span> model for quality prediction.</p>
    <p>Navigate to the "Model Performance" page to see detailed metrics and evaluation results.</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "Prediction":
    st.markdown("<h1 class='main-header'>Quality Prediction Tool</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='info-text'>
    Use this tool to predict the quality class based on process parameters. Adjust the sliders to set the values for each parameter.
    </p>
    """, unsafe_allow_html=True)
    
    # Create two columns for the input parameters
    col1, col2 = st.columns(2)
    
    # Dictionary to store input values
    input_values = {}
    
    # Helper function to create parameter input
    def create_parameter_input(col, feature, info):
        with col:
            input_values[feature] = st.slider(
                f"{feature}",
                min_value=info['min'],
                max_value=info['max'],
                value=info['default'],
                step=info['step'],
                help=f"Range: {info['min']:.2f} - {info['max']:.2f}"
            )
    
    # First column parameters
    create_parameter_input(col1, 'Melt temperature', feature_ranges['Melt temperature'])
    create_parameter_input(col1, 'Mold temperature', feature_ranges['Mold temperature'])
    create_parameter_input(col1, 'time_to_fill', feature_ranges['time_to_fill'])
    create_parameter_input(col1, 'ZDx - Plasticizing time', feature_ranges['ZDx - Plasticizing time'])
    create_parameter_input(col1, 'ZUx - Cycle time', feature_ranges['ZUx - Cycle time'])
    create_parameter_input(col1, 'SKx - Closing force', feature_ranges['SKx - Closing force'])
    create_parameter_input(col1, 'SKs - Clamping force peak value', feature_ranges['SKs - Clamping force peak value'])
    
    # Second column parameters
    create_parameter_input(col2, 'Ms - Torque peak value current cycle', feature_ranges['Ms - Torque peak value current cycle'])
    create_parameter_input(col2, 'Mm - Torque mean value current cycle', feature_ranges['Mm - Torque mean value current cycle'])
    create_parameter_input(col2, 'APSs - Specific back pressure peak value', feature_ranges['APSs - Specific back pressure peak value'])
    create_parameter_input(col2, 'APVs - Specific injection pressure peak value', feature_ranges['APVs - Specific injection pressure peak value'])
    create_parameter_input(col2, 'CPn - Screw position at the end of hold pressure', feature_ranges['CPn - Screw position at the end of hold pressure'])
    create_parameter_input(col2, 'SVo - Shot volume', feature_ranges['SVo - Shot volume'])
    
    # Prediction button
    if st.button("Predict Quality", key="predict_button"):
        with st.spinner("Making prediction..."):
            prediction, probabilities = predict_quality(input_values, best_model, scaler)
            
            # Display the prediction result
            st.markdown("<h2 class='sub-header'>Prediction Result</h2>", unsafe_allow_html=True)
            
            quality_label = get_quality_label(prediction)
            quality_description = get_quality_description(prediction)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Display the predicted class
                st.markdown(f"""
                <div class='prediction-box'>
                    <h3>Predicted Quality Class</h3>
                    <h1 style='color: {"#e53935" if prediction == 0 else "#43a047" if prediction == 2 else "#fb8c00"};'>{quality_label}</h1>
                    <p>{quality_description}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Create a bar chart for class probabilities
                prob_df = pd.DataFrame({
                    'Quality Class': ['Waste', 'Acceptable', 'Target', 'Inefficient'],
                    'Probability': probabilities
                })
                
                fig = px.bar(
                    prob_df, 
                    x='Quality Class', 
                    y='Probability',
                    color='Quality Class',
                    color_discrete_sequence=px.colors.qualitative.Set1,
                    text=prob_df['Probability'].apply(lambda x: f'{x:.2%}')
                )
                
                fig.update_layout(
                    title='Prediction Probabilities',
                    xaxis_title='Quality Class',
                    yaxis_title='Probability',
                    yaxis=dict(tickformat='.0%', range=[0, 1]),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation based on prediction
            st.markdown("<h3>Recommendations</h3>", unsafe_allow_html=True)
            
            if prediction == 0:  # Waste
                st.error("The parameters indicate a high risk of waste production. Consider adjusting the following parameters:")
                # Identify top influential parameters for avoiding waste
                st.markdown("""
                - Decrease injection pressure (APVs)
                - Adjust melt temperature closer to optimal range
                - Increase mold temperature
                - Review cycle time and plasticizing time balance
                """)
            
            elif prediction == 1:  # Acceptable
                st.warning("The product is predicted to be acceptable but not optimal. To achieve target quality, consider:")
                st.markdown("""
                - Fine-tune the melt temperature
                - Adjust the injection pressure
                - Balance the cycle time to optimize quality
                - Review the clamping force parameters
                """)
            
            elif prediction == 2:  # Target
                st.success("The parameters are well-optimized for target quality production. Maintain these settings for consistent quality.")
            
            elif prediction == 3:  # Inefficient
                st.info("The product is predicted to be of good quality but produced inefficiently. Consider:")
                st.markdown("""
                - Optimize cycle time to improve efficiency
                - Evaluate if the injection pressure can be reduced while maintaining quality
                - Review the torque values for potential energy savings
                - Balance plasticizing time for better resource utilization
                """)

elif page == "Model Performance":
    st.markdown("<h1 class='main-header'>Model Performance Analysis</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='info-text'>
    This page displays the performance metrics and evaluation results for the machine learning model
    used for quality prediction.
    </p>
    """, unsafe_allow_html=True)
    
    # Create tabs for different evaluation aspects
    tab1, tab2, tab3 = st.tabs(["Overview", "Confusion Matrix", "Class Metrics"])
    
    with tab1:
        st.markdown("<h2 class='sub-header'>Model Performance Overview</h2>", unsafe_allow_html=True)
        
        # Display model type and key metrics
        st.markdown(f"""
        <div class='metric-card'>
        <h3>Model Information</h3>
        <p><b>Model Type:</b> {best_model_name}</p>
        <p><b>Description:</b> {best_model.__class__.__doc__.split('.')[0] if best_model.__doc__ else 'A machine learning model for quality classification'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a test prediction on the dataset to get metrics
        # We'll use a sample of the dataset for demonstration
        X_sample = df.drop('quality', axis=1).sample(frac=0.3, random_state=42)
        y_sample = df.loc[X_sample.index, 'quality']
        
        # Apply feature engineering
        X_processed = X_sample.copy()
        X_processed['Pressure_Temp_Ratio'] = X_processed['APVs - Specific injection pressure peak value'] / X_processed['Melt temperature']
        X_processed['Efficiency_Ratio'] = X_processed['ZUx - Cycle time'] / X_processed['ZDx - Plasticizing time']
        X_processed['Power_Index'] = X_processed['Ms - Torque peak value current cycle'] * X_processed['Mm - Torque mean value current cycle']
        X_processed['Pressure_Difference'] = X_processed['APVs - Specific injection pressure peak value'] - X_processed['APSs - Specific back pressure peak value']
        X_processed['Temp_Interaction'] = X_processed['Melt temperature'] * X_processed['Mold temperature']
        X_processed['Volume_Fill_Ratio'] = X_processed['SVo - Shot volume'] / X_processed['time_to_fill']
        
        critical_features = ['Melt temperature', 'Mold temperature', 'APVs - Specific injection pressure peak value', 'time_to_fill']
        for feature in critical_features:
            X_processed[f'{feature}_squared'] = X_processed[feature] ** 2
        
        X_processed['Process_Stability'] = ((X_processed['SKs - Clamping force peak value'] / X_processed['SKx - Closing force']) *
                                        (X_processed['Ms - Torque peak value current cycle'] / X_processed['Mm - Torque mean value current cycle']))
        
        # Scale features
        X_scaled = scaler.transform(X_processed)
        
        # Make predictions
        if best_model.__class__.__name__ == 'Sequential':
            y_pred = np.argmax(best_model.predict(X_scaled), axis=1)
        else:
            y_pred = best_model.predict(X_scaled)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_sample, y_pred)
        precision = precision_score(y_sample, y_pred, average='macro')
        recall = recall_score(y_sample, y_pred, average='macro')
        f1 = f1_score(y_sample, y_pred, average='macro')
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}")
        
        with col2:
            st.metric("Precision", f"{precision:.2%}")
        
        with col3:
            st.metric("Recall", f"{recall:.2%}")
        
        with col4:
            st.metric("F1 Score", f"{f1:.2%}")
        
        # Classification report
        st.subheader("Classification Report")
        report = classification_report(y_sample, y_pred, target_names=['Waste', 'Acceptable', 'Target', 'Inefficient'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}").highlight_max(axis=0, subset=['precision', 'recall', 'f1-score']))
    
    with tab2:
        st.markdown("<h2 class='sub-header'>Confusion Matrix</h2>", unsafe_allow_html=True)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_sample, y_pred)
        
        # Display the confusion matrix
        fig = plot_confusion_matrix(cm, classes=['Waste', 'Acceptable', 'Target', 'Inefficient'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation
        st.markdown("""
        <div class='metric-card'>
        <h3>Interpretation</h3>
        <p>The confusion matrix shows how many samples were correctly classified (diagonal) versus misclassified (off-diagonal).</p>
        <p>Each row represents the true class, while each column represents the predicted class.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<h2 class='sub-header'>Class-wise Metrics</h2>", unsafe_allow_html=True)
        
        # Calculate and display class-wise metrics
        class_metrics_df = get_class_metrics(y_sample, y_pred)
        
        # Plot class metrics as a bar chart
        fig = px.bar(
            class_metrics_df.melt(id_vars=['Class'], var_name='Metric', value_name='Value'),
            x='Class',
            y='Value',
            color='Metric',
            barmode='group',
            text_auto='.2f',
            height=500
        )
        
        fig.update_layout(
            title='Class-wise Performance Metrics',
            xaxis_title='Quality Class',
            yaxis_title='Score',
            legend_title='Metric',
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the metrics in a table
        st.subheader("Detailed Class Metrics")
        format_dict = {
    'Precision': "{:.4f}",
    'Recall': "{:.4f}",
    'F1 Score': "{:.4f}"
}

        st.dataframe(
    class_metrics_df.style
    .format(format_dict)
    .highlight_max(axis=0, subset=['Precision', 'Recall', 'F1 Score'])
)
        # Add interpretation
        st.markdown("""
        <div class='metric-card'>
        <h3>Interpretation</h3>
        <p><b>Precision:</b> Percentage of correct positive predictions for each class. Higher precision means fewer false positives.</p>
        <p><b>Recall:</b> Percentage of actual positives that were correctly identified. Higher recall means fewer false negatives.</p>
        <p><b>F1 Score:</b> Harmonic mean of precision and recall, providing a balance between the two metrics.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "Statistical Analysis":
    st.markdown("<h1 class='main-header'>Statistical Analysis</h1>", unsafe_allow_html=True)

    if hypothesis_results is None:
        st.warning("Hypothesis testing results not available.")
    elif 'anova_results' not in hypothesis_results:
        st.error("ANOVA results not found in the loaded data.")
    else:
        anova_dict = hypothesis_results['anova_results']

        # Filter only features that exist in the DataFrame
        valid_features = [key for key in anova_dict.keys() if key in df.columns]

        if not valid_features:
            st.warning("No valid features available for ANOVA plotting.")
        else:
            st.markdown("### ANOVA Box Plots by Feature")
            selected_feature = st.selectbox("Select a feature to view ANOVA results", valid_features)

            fig = plot_anova_results(df, selected_feature)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class='metric-card'>
                <p>This box plot shows the distribution of the selected feature across different quality classes. 
                Use this to identify if there is a significant difference in feature behavior by class.</p>
            </div>
            """, unsafe_allow_html=True)

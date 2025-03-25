import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit Page Configuration
st.set_page_config(
    page_title="Interactive Clustering Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open('custom.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state for data
if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.numeric_data = None
    st.session_state.scaled_features = None

# Sidebar Navigation
with st.sidebar:
    st.title("main")
    
    # Navigation items with icons
    selected = st.radio(
        "",
        ["🏠 Main", "📊 Visualization", "📈 Data Analysis"],
        label_visibility="collapsed"
    )

if selected == "🏠 Main":
    st.title("Interactive Clustering Analysis")
    
    st.write("Welcome to the Interactive Clustering Analysis tool! This application helps you:")
    st.markdown("""
    - Analyze your dataset with multiple clustering algorithms
    - Visualize clustering results interactively
    - Explore detailed data insights
    """)

    # File Upload Section
    st.markdown("### Upload your dataset (CSV or Excel)")
    upload_section = st.container()

    with upload_section:
        uploaded_file = st.file_uploader(
            "",
            type=["csv", "xls", "xlsx"],
            help="Limit 200MB per file • CSV, XLSX, XLS"
        )

        if uploaded_file:
            try:
                # Load and store data in session state
                if uploaded_file.name.endswith(".csv"):
                    try:
                        st.session_state.data = pd.read_csv(uploaded_file, encoding="utf-8")
                    except UnicodeDecodeError:
                        try:
                            st.session_state.data = pd.read_csv(uploaded_file, encoding="latin1")
                        except Exception as e:
                            st.error(f"Error loading file: {str(e)}")
                            st.stop()
                else:
                    try:
                        st.session_state.data = pd.read_excel(uploaded_file)
                    except Exception as e:
                        st.error(f"Error loading Excel file: {str(e)}")
                        st.stop()

                df = st.session_state.data
                
                # Data Overview Section
                st.markdown("### Dataset Overview")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Rows", df.shape[0])
                col2.metric("Total Columns", df.shape[1])
                col3.metric("Missing Values", df.isnull().sum().sum())
                col4.metric("Numeric Columns", len(df.select_dtypes(include=["float64", "int64"]).columns))

                # Data Preview
                st.markdown("### Data Preview")
                st.dataframe(df.head(), use_container_width=True)

                # Data Types Information
                st.markdown("### Data Types Information")
                dtype_df = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Unique Values': df.nunique()
                })
                st.dataframe(dtype_df, use_container_width=True)

                # Store numeric data and scaled features
                numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
                if len(numeric_cols) > 0:
                    st.session_state.numeric_data = df[numeric_cols].copy()
                    scaler = StandardScaler()
                    st.session_state.scaled_features = scaler.fit_transform(st.session_state.numeric_data)
                else:
                    st.error("No numeric columns found in the dataset. Please upload a dataset with numeric columns for clustering analysis.")
                    st.stop()

            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
        else:
            st.warning("⚠ Please upload a dataset to begin your analysis.")

elif selected == "📊 Visualization":
    if st.session_state.data is None:
        st.warning("Please upload a dataset in the Main section first.")
    else:
        st.markdown("## Visualization")
        st.markdown("### Explore clustering patterns with interactive visualizations")
        
        # Clustering Options
        algorithm = st.selectbox("Select Algorithm", ["KMeans", "DBSCAN", "Hierarchical"])
        
        col1, col2 = st.columns(2)
        with col1:
            if algorithm == "KMeans":
                n_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=3)
                model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            elif algorithm == "DBSCAN":
                eps = st.slider("Epsilon (ε)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
                min_samples = st.slider("Min Samples", min_value=2, max_value=20, value=5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
            else:
                n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
                linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

        # Perform clustering
        labels = model.fit_predict(st.session_state.scaled_features)

        # Visualization
        st.markdown("### Clustering Results")
        
        # Feature selection for visualization
        numeric_cols = st.session_state.numeric_data.columns
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("Select X-axis feature", numeric_cols, index=0)
        with col2:
            y_feature = st.selectbox("Select Y-axis feature", numeric_cols, index=min(1, len(numeric_cols)-1))
        
        # Create scatter plot
        fig = px.scatter(
            st.session_state.numeric_data,
            x=x_feature,
            y=y_feature,
            color=labels.astype(str),
            title=f"{algorithm} Clustering Results",
            labels={"color": "Cluster"},
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display Cluster Metrics
        st.markdown("### Clustering Metrics")
        metric_cols = st.columns(2)
        if len(set(labels)) > 1:
            metric_cols[0].metric("Silhouette Score", f"{silhouette_score(st.session_state.scaled_features, labels):.3f}")
            metric_cols[1].metric("Calinski-Harabasz Score", f"{calinski_harabasz_score(st.session_state.scaled_features, labels):.3f}")
        else:
            st.warning("Cannot compute metrics for a single cluster.")

elif selected == "📈 Data Analysis":
    if st.session_state.data is None:
        st.warning("Please upload a dataset in the Main section first.")
    else:
        st.markdown("## Exploratory Data Analysis")
        
        # Select analysis type
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Numerical Analysis", "Distribution Analysis", "Correlation Analysis", "Missing Values Analysis"]
        )
        
        if analysis_type == "Numerical Analysis":
            st.markdown("### Numerical Features Summary")
            numeric_df = st.session_state.numeric_data
            
            # Summary statistics
            st.dataframe(numeric_df.describe(), use_container_width=True)
            
            # Box plots for numerical columns
            st.markdown("### Box Plots")
            selected_columns = st.multiselect(
                "Select columns for box plot",
                numeric_df.columns.tolist(),
                default=numeric_df.columns[:min(5, len(numeric_df.columns))].tolist()
            )
            if selected_columns:
                fig = px.box(numeric_df[selected_columns], title="Distribution of Numerical Features")
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Distribution Analysis":
            st.markdown("### Distribution Analysis")
            
            # Select column for histogram
            column = st.selectbox("Select column for distribution analysis", st.session_state.numeric_data.columns)
            
            col1, col2 = st.columns(2)
            with col1:
                # Histogram
                fig_hist = px.histogram(
                    st.session_state.numeric_data,
                    x=column,
                    title=f"Histogram of {column}",
                    template="plotly_white"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # KDE plot
                fig_kde = px.line(
                    y=st.session_state.numeric_data[column].value_counts(normalize=True, bins=30),
                    title=f"Density Plot of {column}",
                    template="plotly_white"
                )
                st.plotly_chart(fig_kde, use_container_width=True)
        
        elif analysis_type == "Correlation Analysis":
            st.markdown("### Correlation Analysis")
            
            # Correlation matrix
            correlation = st.session_state.numeric_data.corr()
            
            # Heatmap
            fig = px.imshow(
                correlation,
                title="Feature Correlation Matrix",
                template="plotly_white",
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed correlation values
            st.markdown("### Correlation Values")
            st.dataframe(correlation.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1))
        
        elif analysis_type == "Missing Values Analysis":
            st.markdown("### Missing Values Analysis")
            
            # Calculate missing values
            missing_df = pd.DataFrame({
                'Column': st.session_state.data.columns,
                'Missing Values': st.session_state.data.isnull().sum(),
                'Percentage': (st.session_state.data.isnull().sum() / len(st.session_state.data) * 100).round(2)
            }).sort_values('Percentage', ascending=False)
            
            # Display missing values table
            st.dataframe(missing_df, use_container_width=True)
            
            # Missing values visualization
            fig = px.bar(
                missing_df,
                x='Column',
                y='Percentage',
                title='Missing Values Percentage by Column',
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

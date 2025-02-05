import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import calendar
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import calendar

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
HOSTEL_DAILY_LIMITS = {
    'Large': 250000,  # 250 kL per day
    'Medium': 120000, # 120 kL per day
    'Small': 60000,   # 60 kL per day
}

CAMPUS_DAILY_LIMIT = 6.0  # 6 MLD


# Add these constants for water quality thresholds
WATER_QUALITY_LIMITS = {
    'COD': {'normal': 250, 'warning': 350, 'critical': 500},  # mg/L
    'BOD': {'normal': 30, 'warning': 50, 'critical': 100},    # mg/L
    'TSS': {'normal': 100, 'warning': 150, 'critical': 200}   # mg/L
}

def generate_water_quality_data():
    """Generate synthetic water quality data"""
    dates = pd.date_range(start='2023-02-01', end='2024-02-05', freq='H')
    
    data = {
        'Date': dates,
        'COD': [],
        'BOD': [],
        'TSS': []
    }
    
    # Generate realistic patterns with daily and seasonal variations
    for i, date in enumerate(dates):
        # Base values
        cod_base = 200 + np.sin(i/24 * np.pi) * 20  # Daily pattern
        bod_base = 25 + np.sin(i/24 * np.pi) * 5
        tss_base = 80 + np.sin(i/24 * np.pi) * 10
        
        # Add seasonal variation
        seasonal_factor = 1 + np.sin(i/(24*365) * 2 * np.pi) * 0.2
        
        # Add random events and noise
        event_factor = 1.0
        if np.random.random() < 0.02:  # Random events
            event_factor = np.random.uniform(1.2, 2.3)
        
        noise = np.random.normal(0, 0.1)
        
        data['COD'].append(cod_base * seasonal_factor * event_factor * (1 + noise))
        data['BOD'].append(bod_base * seasonal_factor * event_factor * (1 + noise))
        data['TSS'].append(tss_base * seasonal_factor * event_factor * (1 + noise))
    
    return pd.DataFrame(data)

def analyze_water_quality(df_quality):
    """Analyze water quality data and detect violations"""
    analysis = df_quality.copy()
    
    # Calculate status for each parameter
    for param in ['COD', 'BOD', 'TSS']:
        analysis[f'{param}_status'] = np.where(
            analysis[param] > WATER_QUALITY_LIMITS[param]['critical'], 'Critical',
            np.where(analysis[param] > WATER_QUALITY_LIMITS[param]['warning'], 'Warning', 'Normal')
        )
    
    # Calculate violation durations
    violations = {param: [] for param in ['COD', 'BOD', 'TSS']}
    for param in violations:
        mask = analysis[f'{param}_status'] != 'Normal'
        violation_groups = mask.ne(mask.shift()).cumsum()[mask]
        violations[param] = violation_groups.value_counts()
    
    return analysis, violations

def predict_water_quality(df_quality, forecast_hours=24):
    """Predict future water quality parameters using ML"""
    predictions = {}
    confidence_intervals = {}
    
    for param in ['COD', 'BOD', 'TSS']:
        # Prepare features
        df = df_quality.copy()
        df['hour'] = df['Date'].dt.hour
        df['day'] = df['Date'].dt.day
        df['month'] = df['Date'].dt.month
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        X = df[['hour', 'day', 'month']].values
        y = df[param].values
        
        model.fit(X, y)
        
        # Generate future dates
        future_dates = pd.date_range(
            start=df_quality['Date'].max(),
            periods=forecast_hours,
            freq='H'
        )
        
        # Prepare future features
        future_X = pd.DataFrame({
            'hour': future_dates.hour,
            'day': future_dates.day,
            'month': future_dates.month
        }).values
        
        # Make predictions
        pred = model.predict(future_X)
        predictions[param] = pred
        
        # Calculate confidence intervals using prediction std
        pred_std = np.std([
            tree.predict(future_X)
            for tree in model.estimators_
        ], axis=0)
        
        confidence_intervals[param] = {
            'upper': pred + 1.96 * pred_std,
            'lower': pred - 1.96 * pred_std
        }
    
    return predictions, confidence_intervals, future_dates

def create_water_quality_section(df_quality, quality_analysis, violations, predictions, confidence_intervals, future_dates):
    """Create water quality monitoring section"""
    st.subheader("Water Quality Monitoring")
    
    # Current status metrics
    col1, col2, col3 = st.columns(3)
    latest = df_quality.iloc[-1]
    
    for i, param in enumerate(['COD', 'BOD', 'TSS']):
        with [col1, col2, col3][i]:
            current_value = latest[param]
            status = quality_analysis[f'{param}_status'].iloc[-1]
            
            st.metric(
                f"{param} Level",
                f"{current_value:.1f} mg/L",
                f"Limit: {WATER_QUALITY_LIMITS[param]['normal']} mg/L"
            )
            
            if status == 'Critical':
                st.error(f"⚠️ Critical {param} Level!")
            elif status == 'Warning':
                st.warning(f"⚡ High {param} Level")
            else:
                st.success(f"✅ Normal {param} Level")
    
    # Historical trends
    st.subheader("Parameter Trends")
    
    # Create multi-parameter plot
    fig = go.Figure()
    
    for param in ['COD', 'BOD', 'TSS']:
        # Historical data
        fig.add_trace(go.Scatter(
            x=df_quality['Date'],
            y=df_quality[param],
            name=f"{param} Actual"
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions[param],
            name=f"{param} Predicted",
            line=dict(dash='dash')
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=confidence_intervals[param]['upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=confidence_intervals[param]['lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name=f"{param} Confidence Interval"
        ))
        
        # Add threshold lines
        for level, value in WATER_QUALITY_LIMITS[param].items():
            fig.add_hline(
                y=value,
                line_dash="dash",
                annotation_text=f"{param} {level} limit",
                line_color="red" if level == "critical" else "orange" if level == "warning" else "green"
            )
    
    fig.update_layout(
        title='Water Quality Parameters - Historical Data and Predictions',
        height=600,
        yaxis_title='Concentration (mg/L)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Violation analysis
    st.subheader("Quality Violations Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Violation counts
        violation_data = pd.DataFrame({
            'Parameter': ['COD', 'BOD', 'TSS'],
            'Warning': [
                (quality_analysis[f'{param}_status'] == 'Warning').sum()
                for param in ['COD', 'BOD', 'TSS']
            ],
            'Critical': [
                (quality_analysis[f'{param}_status'] == 'Critical').sum()
                for param in ['COD', 'BOD', 'TSS']
            ]
        })
        
        fig = px.bar(
            violation_data,
            x='Parameter',
            y=['Warning', 'Critical'],
            title='Quality Violations by Parameter',
            barmode='stack'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Violation duration distribution
        for param in ['COD', 'BOD', 'TSS']:
            if len(violations[param]) > 0:
                st.write(f"**{param} Violation Durations:**")
                st.write(f"- Average: {violations[param].mean():.1f} hours")
                st.write(f"- Maximum: {violations[param].max():.1f} hours")

def create_treatment_optimization_section(df_quality):
    """Create treatment process optimization section"""
    st.subheader("Treatment Process Optimization")
    
    # Calculate correlations
    correlations = df_quality[['COD', 'BOD', 'TSS']].corr()
    
    # Process efficiency metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cod_removal = (1 - df_quality['COD'].mean() / WATER_QUALITY_LIMITS['COD']['normal']) * 100
        st.metric("COD Removal Efficiency", f"{cod_removal:.1f}%")
    
    with col2:
        bod_removal = (1 - df_quality['BOD'].mean() / WATER_QUALITY_LIMITS['BOD']['normal']) * 100
        st.metric("BOD Removal Efficiency", f"{bod_removal:.1f}%")
    
    with col3:
        tss_removal = (1 - df_quality['TSS'].mean() / WATER_QUALITY_LIMITS['TSS']['normal']) * 100
        st.metric("TSS Removal Efficiency", f"{tss_removal:.1f}%")
    
    # Parameter relationships
    st.subheader("Parameter Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation heatmap
        fig = px.imshow(
            correlations,
            labels=dict(color="Correlation"),
            title="Parameter Correlations"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Replace scatter matrix with paired scatter plots
        # Create subplots for parameter pairs
        fig = make_subplots(rows=2, cols=2)
        
        # COD vs BOD
        fig.add_trace(
            go.Scatter(
                x=df_quality['COD'],
                y=df_quality['BOD'],
                mode='markers',
                name='COD vs BOD'
            ),
            row=1, col=1
        )
        
        # COD vs TSS
        fig.add_trace(
            go.Scatter(
                x=df_quality['COD'],
                y=df_quality['TSS'],
                mode='markers',
                name='COD vs TSS'
            ),
            row=1, col=2
        )
        
        # BOD vs TSS
        fig.add_trace(
            go.Scatter(
                x=df_quality['BOD'],
                y=df_quality['TSS'],
                mode='markers',
                name='BOD vs TSS'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=400,
            title="Parameter Relationships",
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="COD (mg/L)", row=1, col=1)
        fig.update_xaxes(title_text="COD (mg/L)", row=1, col=2)
        fig.update_xaxes(title_text="BOD (mg/L)", row=2, col=1)
        
        fig.update_yaxes(title_text="BOD (mg/L)", row=1, col=1)
        fig.update_yaxes(title_text="TSS (mg/L)", row=1, col=2)
        fig.update_yaxes(title_text="TSS (mg/L)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Treatment recommendations
    st.subheader("Treatment Recommendations")
    
    # Calculate current values and trends
    latest_values = df_quality.iloc[-1]
    trends = df_quality.tail(24).mean() - df_quality.tail(48).head(24).mean()
    
    # Create a more structured recommendations system
    st.write("### Current Status and Recommendations")
    
    for param in ['COD', 'BOD', 'TSS']:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if latest_values[param] > WATER_QUALITY_LIMITS[param]['critical']:
                st.error(f"⚠️ {param}: Critical")
            elif latest_values[param] > WATER_QUALITY_LIMITS[param]['warning']:
                st.warning(f"⚡ {param}: Warning")
            else:
                st.success(f"✅ {param}: Normal")
        
        with col2:
            recommendations = []
            
            if latest_values[param] > WATER_QUALITY_LIMITS[param]['warning']:
                if param == 'COD':
                    recommendations.extend([
                        "Optimize chemical dosing system",
                        "Check aeration system efficiency",
                        "Verify retention time in treatment units"
                    ])
                elif param == 'BOD':
                    recommendations.extend([
                        "Monitor biological treatment system",
                        "Check nutrient levels and ratios",
                        "Verify dissolved oxygen levels"
                    ])
                else:  # TSS
                    recommendations.extend([
                        "Inspect filtration system",
                        "Optimize coagulation/flocculation process",
                        "Check settling tank performance"
                    ])
            
            if trends[param] > 0:
                recommendations.append(f"Monitor rising trend: +{trends[param]:.1f} mg/L over 24h")
            
            if recommendations:
                for rec in recommendations:
                    st.write(f"- {rec}")
            else:
                st.write("- Maintain current treatment parameters")
    
    # Add overall system performance summary
    st.write("### System Performance Summary")
    avg_efficiency = (cod_removal + bod_removal + tss_removal) / 3
    
    if avg_efficiency > 90:
        st.success(f"🌟 Overall system efficiency is excellent: {avg_efficiency:.1f}%")
    elif avg_efficiency > 75:
        st.info(f"📊 Overall system efficiency is good: {avg_efficiency:.1f}%")
    else:
        st.warning(f"⚡ Overall system efficiency needs improvement: {avg_efficiency:.1f}%")
def generate_synthetic_data():
    dates = pd.date_range(start='2023-02-01', end='2024-02-05', freq='D')
    
    # Base patterns with random variations
    base_demand = 4.4
    base_release = 5.2
    
    # Generate more realistic patterns
    data = {
        'Date': dates,
        'WTP_Release': [],
        'Total_Demand': []
    }
    
    # Add random events and gradual changes
    for i, date in enumerate(dates):
        # Add random events
        event_factor = 1.0
        if np.random.random() < 0.05:  # Random events 5% of the time
            event_factor = np.random.choice([0.8, 1.2])  # Significant changes
            
        # Add gradual changes
        trend = 1 + (i / len(dates)) * 0.1  # 10% increase over the period
        
        # Daily variation (more on weekdays)
        day_factor = 1.1 if date.dayofweek < 5 else 0.9
        
        # Random noise
        noise = np.random.normal(0, 0.05)
        
        demand = base_demand * event_factor * trend * day_factor * (1 + noise)
        # Release is usually higher than demand
        release = base_release * event_factor * trend * day_factor * (1 + np.random.normal(0, 0.07))
        
        data['Total_Demand'].append(demand)
        data['WTP_Release'].append(release)
    
    df = pd.DataFrame(data)
    df['Wastage'] = df['WTP_Release'] - df['Total_Demand']
    
    return df

# Enhanced hostel data generation
def generate_hostel_data():
    hostels = {
        'Large': {
            'hostels': ['Brahmaputra', 'Lohit', 'Disang'],
            'base_demand': 234000,
            'occupancy': 800
        },
        'Medium': {
            'hostels': ['Barak', 'Umiam', 'Kameng', 'Subansiri', 'Dhansiri', 'Married Scholars'],
            'base_demand': 108000,
            'occupancy': 400
        },
        'Small': {
            'hostels': ['Kapili', 'Manas', 'Dihing', 'Siang'],
            'base_demand': 54000,
            'occupancy': 200
        }
    }
    
    dates = pd.date_range(start='2024-01-01', end='2024-02-05', freq='D')
    data = []
    
    for size, info in hostels.items():
        for hostel in info['hostels']:
            # Generate unique pattern for each hostel
            base_variation = np.random.normal(1, 0.05)  # Each hostel has slightly different base usage
            
            for date in dates:
                # More realistic variations
                time_variation = 1.2 if 6 <= date.hour <= 20 else 0.8  # Day/night variation
                weekday_variation = 1.1 if date.dayofweek < 5 else 0.9  # Weekday/weekend
                occupancy_factor = np.random.normal(0.95, 0.05)  # Random occupancy variations
                
                # Random events (parties, events, etc.)
                event_factor = 1.0
                if np.random.random() < 0.02:  # 2% chance of events
                    event_factor = np.random.uniform(1.2, 1.4)
                
                demand = (info['base_demand'] * base_variation * time_variation * 
                         weekday_variation * occupancy_factor * event_factor)
                
                data.append({
                    'Date': date,
                    'Hostel': hostel,
                    'Size': size,
                    'Demand': demand,
                    'Capacity': info['base_demand'],
                    'Occupancy': info['occupancy'] * occupancy_factor,
                    'PerPersonUsage': demand / (info['occupancy'] * occupancy_factor)
                })
    
    return pd.DataFrame(data)

# Improved anomaly detection
from sklearn.ensemble import IsolationForest
import pandas as pd

def detect_campus_anomalies(df_main, df_hostels):
    # Use Isolation Forest for anomaly detection
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    features = df_main[['Total_Demand', 'WTP_Release', 'Wastage']].copy()
    anomalies = iso_forest.fit_predict(features)
    anomaly_scores = iso_forest.score_samples(features)
    
    df_anomalies = df_main.copy()
    df_anomalies['is_anomaly'] = anomalies == -1
    df_anomalies['anomaly_score'] = anomaly_scores
    
    # Add location detection for anomalies
    for idx in df_anomalies[df_anomalies['is_anomaly']].index:
        date = df_anomalies.loc[idx, 'Date']
        
        # Check hostels for anomalies on the same date
        hostel_usage = df_hostels[df_hostels['Date'] == date].copy()
        
        # Ensure hostel_limits is correctly mapped
        hostel_limits = hostel_usage['Size'].map(HOSTEL_DAILY_LIMITS)

        # Ensure both columns are numeric and aligned
        hostel_usage['Demand'] = pd.to_numeric(hostel_usage['Demand'], errors='coerce')
        hostel_limits = pd.to_numeric(hostel_limits, errors='coerce')
        
        # Align indexes before performing operations
        hostel_usage, hostel_limits = hostel_usage.align(hostel_limits, axis=0, copy=False)

        # Find anomalous hostels
        anomalous_hostels = hostel_usage.loc[
            hostel_usage['Demand'] > hostel_limits * 0.9, 'Hostel'
        ].tolist()
        
        # Assign location based on anomalies
        if anomalous_hostels:
            df_anomalies.loc[idx, 'location'] = f"Hostels: {', '.join(anomalous_hostels)}"
        else:
            df_anomalies.loc[idx, 'location'] = "Main Distribution System"
    
    return df_anomalies


# Enhanced hostel monitoring
def analyze_hostel_efficiency(df_hostels):
    latest_date = df_hostels['Date'].max()
    current_usage = df_hostels[df_hostels['Date'] == latest_date].copy()
    
    # Calculate efficiency metrics
    current_usage['Efficiency'] = (current_usage['Capacity'] - current_usage['Demand']) / current_usage['Capacity'] * 100
    current_usage['PerPersonUsage'] = current_usage['Demand'] / current_usage['Occupancy']
    
    # Identify problematic hostels
    current_usage['Status'] = np.where(current_usage['Efficiency'] < 0, 'Over Capacity',
                              np.where(current_usage['Efficiency'] < 10, 'Near Capacity', 'Normal'))
    
    return current_usage

# Predict future requirements
def predict_future_requirements(df_main, df_hostels, days=120):
    # Aggregate historical data
    historical_data = {
        'campus': df_main['Total_Demand'].values,
        'hostels': df_hostels.groupby('Date')['Demand'].sum().values
    }
    
    predictions = {}
    confidence_intervals = {}
    
    for location, data in historical_data.items():
        # Calculate trend
        x = np.arange(len(data))
        z = np.polyfit(x, data, 1)
        trend = np.poly1d(z)
        
        # Predict future values
        future_x = np.arange(len(data), len(data) + days)
        future_trend = trend(future_x)
        
        # Add random variations
        variations = np.random.normal(0, np.std(data) * 0.5, days)
        predictions[location] = future_trend + variations
        
        # Calculate confidence intervals
        std_dev = np.std(data)
        confidence_intervals[location] = {
            'upper': future_trend + 1.96 * std_dev,
            'lower': future_trend - 1.96 * std_dev
        }
    
    return predictions, confidence_intervals

def create_overview_section(df_main, df_hostels):
    st.subheader("Campus Overview")
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_demand = df_main['Total_Demand'].iloc[-1]
        limit_percentage = (current_demand / CAMPUS_DAILY_LIMIT) * 100
        st.metric("Current Demand", f"{current_demand:.2f} MLD",
                 f"{limit_percentage:.1f}% of limit")
        if limit_percentage > 90:
            st.error("⚠️ Near daily limit!")
    with col2:
        current_wastage = df_main['Wastage'].iloc[-1]
        st.metric("Current Wastage", f"{current_wastage:.2f} MLD",
                 f"{current_wastage - df_main['Wastage'].iloc[-2]:.2f} MLD")
    with col3:
        efficiency = ((1 - current_wastage/df_main['WTP_Release'].iloc[-1]) * 100)
        st.metric("System Efficiency", f"{efficiency:.1f}%")
    with col4:
        total_consumption = df_hostels[df_hostels['Date'] == df_hostels['Date'].max()]['Demand'].sum()
        st.metric("Total Hostel Consumption", f"{total_consumption/1000:.1f} kLD")

    # Create two rows of charts
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        # Overall usage trend
        fig = px.line(df_main, x='Date', y=['Total_Demand', 'WTP_Release'],
                     title='Campus Water Usage Trend')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with row1_col2:
        st.subheader("Daily Limit Monitoring")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_main['Date'],
            y=df_main['Total_Demand'],
            name='Actual Usage'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_main['Date'],
            y=[CAMPUS_DAILY_LIMIT] * len(df_main),
            name='Daily Limit',
            line=dict(dash='dash', color='red')
        ))
        
        fig.update_layout(
            title='Campus Usage vs Daily Limit',
            height=400,
            yaxis_title='Water Usage (MLD)'
        )
        st.plotly_chart(fig, use_container_width=True)
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        # Weekly pattern
        weekly_pattern = df_main.copy()
        weekly_pattern['Weekday'] = weekly_pattern['Date'].dt.day_name()
        weekly_avg = weekly_pattern.groupby('Weekday')['Total_Demand'].mean()
        
        fig = px.bar(x=weekly_avg.index, y=weekly_avg.values,
                    title='Average Weekly Usage Pattern',
                    labels={'x': 'Day of Week', 'y': 'Average Demand (MLD)'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with row2_col2:
        # Wastage analysis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=df_main['Date'], y=df_main['Wastage'],
                      name="Daily Wastage"),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=df_main['Date'], 
                      y=df_main['Wastage'].rolling(7).mean(),
                      name="7-day Moving Average"),
            secondary_y=True
        )
        
        fig.update_layout(title='Water Wastage Analysis',
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
    
def create_hostel_section(df_hostels, hostel_analysis):
    st.subheader("Hostel Water Usage Analysis")
    
    # Hostel selection with size filter
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_size = st.selectbox("Filter by Size", ['All'] + list(df_hostels['Size'].unique()))
    with col2:
        if selected_size == 'All':
            hostel_options = df_hostels['Hostel'].unique()
        else:
            hostel_options = df_hostels[df_hostels['Size'] == selected_size]['Hostel'].unique()
        selected_hostel = st.selectbox("Select Hostel", hostel_options)

    # Detailed hostel analysis
    hostel_data = df_hostels[df_hostels['Hostel'] == selected_hostel]
    
    col1, col2, col3, col4 = st.columns(4)
    current_metrics = hostel_analysis[hostel_analysis['Hostel'] == selected_hostel].iloc[0]
    hostel_limit = HOSTEL_DAILY_LIMITS[current_metrics['Size']]
    
    with col1:
        current_usage = current_metrics['Demand']
        limit_percentage = (current_usage / hostel_limit) * 100
        st.metric("Current Usage", f"{current_usage/1000:.1f} kLD",
                 f"{limit_percentage:.1f}% of limit")
        if limit_percentage > 90:
            st.error("⚠️ Near daily limit!")
    with col2:
        st.metric("Per Person Usage", f"{current_metrics['PerPersonUsage']:.1f} L")
    with col3:
        st.metric("Efficiency", f"{current_metrics['Efficiency']:.1f}%")
    with col4:
        st.metric("Current Occupancy", f"{int(current_metrics['Occupancy'])} people")

    # Create three rows of charts
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        # Usage trend
        fig = px.line(hostel_data, x='Date', y='Demand',
                     title=f'{selected_hostel} Usage Trend')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with row1_col2:
        # Per person usage comparison
        fig = px.bar(hostel_analysis, x='Hostel', y='PerPersonUsage',
                    color='Size',
                    title='Per Person Usage Comparison')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        # Efficiency ranking
        fig = px.bar(hostel_analysis.sort_values('Efficiency'),
                    x='Hostel', y='Efficiency',
                    color='Status',
                    title='Hostel Efficiency Ranking')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with row2_col2:
        # Usage distribution by size
        fig = px.box(df_hostels, x='Size', y='Demand',
                    title='Usage Distribution by Hostel Size')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Additional analysis
    st.subheader("Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Daily Limit Monitoring")
        hostel_data = df_hostels[df_hostels['Hostel'] == selected_hostel].copy()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hostel_data['Date'],
            y=hostel_data['Demand'],
            name='Actual Usage'
        ))
        
        fig.add_trace(go.Scatter(
            x=hostel_data['Date'],
            y=[HOSTEL_DAILY_LIMITS[current_metrics['Size']]] * len(hostel_data),
            name='Daily Limit',
            line=dict(dash='dash', color='red')
        ))
        
        fig.update_layout(
            title=f'{selected_hostel} Usage vs Daily Limit',
            height=400,
            yaxis_title='Water Usage (L)'
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # Weekly pattern
        weekly_pattern = hostel_data.copy()
        weekly_pattern['Weekday'] = weekly_pattern['Date'].dt.day_name()
        weekly_avg = weekly_pattern.groupby('Weekday')['Demand'].mean()
        
        fig = px.bar(x=weekly_avg.index, y=weekly_avg.values,
                    title=f'{selected_hostel} - Weekly Usage Pattern',
                    labels={'x': 'Day of Week', 'y': 'Demand (L)'})
        st.plotly_chart(fig, use_container_width=True)
    
def create_anomaly_section(df_anomalies):
    st.subheader("Anomaly Detection")
    
    # Main anomaly plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_anomalies[~df_anomalies['is_anomaly']]['Date'],
        y=df_anomalies[~df_anomalies['is_anomaly']]['Total_Demand'],
        mode='lines',
        name='Normal Usage'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_anomalies[df_anomalies['is_anomaly']]['Date'],
        y=df_anomalies[df_anomalies['is_anomaly']]['Total_Demand'],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=10)
    ))
    
    fig.update_layout(title='Detected Anomalies in Water Usage',
                     height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Additional anomaly analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly distribution
        fig = px.histogram(df_anomalies, x='anomaly_score',
                         color='is_anomaly',
                         title='Distribution of Anomaly Scores')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Wastage vs Demand scatter
        fig = px.scatter(df_anomalies, x='Total_Demand', y='Wastage',
                        color='is_anomaly',
                        title='Wastage vs Demand (Anomalies Highlighted)')
        st.plotly_chart(fig, use_container_width=True)

    # Anomaly details
     
    if df_anomalies['is_anomaly'].any():
        st.subheader("Recent Anomalies")
        
        # Modified anomaly table with location
        recent_anomalies = df_anomalies[df_anomalies['is_anomaly']].tail(7)
        anomaly_table = recent_anomalies[['Date', 'Total_Demand', 'Wastage', 'anomaly_score', 'location']].copy()
        anomaly_table.columns = ['Date', 'Demand (MLD)', 'Wastage (MLD)', 'Severity Score', 'Location']
        st.table(anomaly_table)
        
        # Add location-based analysis
        st.subheader("Anomaly Locations")
        location_counts = recent_anomalies['location'].value_counts()
        
        fig = px.pie(values=location_counts.values, 
                    names=location_counts.index,
                    title='Anomaly Distribution by Location')
        st.plotly_chart(fig, use_container_width=True)
def create_prediction_section(df_main, df_hostels, predictions, confidence_intervals):
    st.subheader("Future Water Requirements Prediction")
    
    # Main prediction plot
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df_main['Date'],
        y=df_main['Total_Demand'],
        name='Historical Demand'
    ))
    
    # Predictions
    future_dates = pd.date_range(
        start=df_main['Date'].max() + timedelta(days=1),
        periods=30,
        freq='D'
    )
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions['campus'],
        name='Predicted Demand',
        line=dict(dash='dash')
    ))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=confidence_intervals['campus']['upper'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=confidence_intervals['campus']['lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='95% Confidence Interval'
    ))
    
    fig.update_layout(title='Water Demand Forecast',
                     height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Additional prediction analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly prediction summary
        monthly_pred = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Demand': predictions['campus']
        })
        monthly_pred['Month'] = monthly_pred['Date'].dt.strftime('%B')
        monthly_avg = monthly_pred.groupby('Month')['Predicted_Demand'].mean()
        
        fig = px.bar(x=monthly_avg.index, y=monthly_avg.values,
                    title='Predicted Monthly Average Demand',
                    labels={'x': 'Month', 'y': 'Predicted Demand (MLD)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Prediction confidence analysis
        conf_width = confidence_intervals['campus']['upper'] - confidence_intervals['campus']['lower']
        fig = px.line(x=future_dates, y=conf_width,
                     title='Prediction Uncertainty Over Time',
                     labels={'x': 'Date', 'y': 'Confidence Interval Width'})
        st.plotly_chart(fig, use_container_width=True)

    # Prediction insights
    st.subheader("Prediction Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_predicted_demand = np.mean(predictions['campus'])
        st.metric("Average Predicted Demand", f"{avg_predicted_demand:.2f} MLD")
    
    with col2:
        max_predicted_demand = np.max(predictions['campus'])
        st.metric("Peak Predicted Demand", f"{max_predicted_demand:.2f} MLD")
    
    with col3:
        current_capacity = df_main['WTP_Release'].max()
        st.metric("Current Capacity", f"{current_capacity:.2f} MLD")

    # Capacity analysis
    if avg_predicted_demand > current_capacity:
        st.error(f"""
            ⚠️ Predicted demand ({avg_predicted_demand:.2f} MLD) exceeds current capacity ({current_capacity:.2f} MLD)
            
            Critical Actions Required:
            1. Plan for immediate capacity expansion
            2. Implement strict conservation measures
            3. Review and optimize large consumption areas
            4. Consider temporary usage restrictions
            5. Accelerate leak detection and repair programs
            
            Risk Level: HIGH
            Time to Capacity: {((current_capacity - avg_predicted_demand) / avg_predicted_demand * 30):.1f} days
        """)
    elif avg_predicted_demand > current_capacity * 0.9:
        st.warning(f"""
            ⚡ Predicted demand ({avg_predicted_demand:.2f} MLD) approaching capacity ({current_capacity:.2f} MLD)
            
            Recommended Actions:
            1. Begin capacity expansion planning
            2. Enhance conservation measures
            3. Review high-usage areas
            4. Increase monitoring frequency
            5. Prepare contingency plans
            
            Risk Level: MEDIUM
            Capacity Headroom: {((current_capacity - avg_predicted_demand) / current_capacity * 100):.1f}%
        """)
    else:
        st.success(f"""
            ✅ Predicted demand ({avg_predicted_demand:.2f} MLD) well within capacity ({current_capacity:.2f} MLD)
            
            Maintenance Actions:
            1. Continue regular monitoring
            2. Maintain conservation measures
            3. Plan for seasonal variations
            4. Update emergency protocols
            5. Consider efficiency improvements
            
            Risk Level: LOW
            Capacity Utilization: {(avg_predicted_demand / current_capacity * 100):.1f}%
        """)

    # Seasonal analysis
    st.subheader("Seasonal Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Historical seasonal pattern
        df_main['Month'] = df_main['Date'].dt.strftime('%B')
        monthly_avg = df_main.groupby('Month')['Total_Demand'].mean()
        
        fig = px.line(x=monthly_avg.index, y=monthly_avg.values,
                     title='Historical Monthly Usage Pattern',
                     labels={'x': 'Month', 'y': 'Average Demand (MLD)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Seasonal forecast
        seasonal_forecast = pd.DataFrame({
            'Month': future_dates.strftime('%B'),
            'Predicted_Demand': predictions['campus']
        })
        monthly_forecast = seasonal_forecast.groupby('Month')['Predicted_Demand'].mean()
        
        fig = px.bar(x=monthly_forecast.index, y=monthly_forecast.values,
                    title='Monthly Demand Forecast',
                    labels={'x': 'Month', 'y': 'Predicted Demand (MLD)'})
        st.plotly_chart(fig, use_container_width=True)

def create_maintenance_section(df_main, df_anomalies):
    st.subheader("Maintenance Dashboard")
    
    # System health metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        efficiency = ((1 - df_main['Wastage'].mean()/df_main['WTP_Release'].mean()) * 100)
        st.metric("Average System Efficiency", f"{efficiency:.1f}%")
    
    with col2:
        anomaly_rate = (df_anomalies['is_anomaly'].sum() / len(df_anomalies)) * 100
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
    
    with col3:
        avg_wastage = df_main['Wastage'].mean()
        st.metric("Average Daily Wastage", f"{avg_wastage:.2f} MLD")
    
    with col4:
        peak_demand = df_main['Total_Demand'].max()
        st.metric("Peak Demand", f"{peak_demand:.2f} MLD")

    # Maintenance analysis charts
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        # System efficiency trend
        efficiency_data = df_main.copy()
        efficiency_data['Efficiency'] = (1 - efficiency_data['Wastage']/efficiency_data['WTP_Release']) * 100
        
        fig = px.line(efficiency_data, x='Date', y='Efficiency',
                     title='System Efficiency Trend')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with row1_col2:
        # Peak demand analysis
        daily_peak = df_main.resample('D', on='Date')['Total_Demand'].max()
        
        fig = px.line(x=daily_peak.index, y=daily_peak.values,
                     title='Daily Peak Demand Trend')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Maintenance recommendations
    st.subheader("System Maintenance Recommendations")
    
    # Calculate various indicators
    recent_efficiency = efficiency_data['Efficiency'].tail(7).mean()
    efficiency_trend = efficiency_data['Efficiency'].tail(30).mean() - efficiency_data['Efficiency'].tail(7).mean()
    recent_anomalies = df_anomalies[df_anomalies['is_anomaly']].tail(7)
    
    # Generate recommendations based on indicators
    if recent_efficiency < 85 or efficiency_trend < -2 or len(recent_anomalies) > 3:
        st.error("""
            🚨 Immediate Maintenance Required
            
            Critical Issues:
            1. System efficiency below target
            2. Multiple anomalies detected
            3. Negative efficiency trend
            
            Recommended Actions:
            - Schedule immediate system inspection
            - Check for leaks in main distribution lines
            - Verify meter calibrations
            - Review pump performance
            - Conduct pressure tests
        """)
    elif recent_efficiency < 90 or efficiency_trend < -1 or len(recent_anomalies) > 1:
        st.warning("""
            ⚠️ Maintenance Review Needed
            
            Issues to Address:
            1. System efficiency slightly below target
            2. Some anomalies detected
            3. Slight negative efficiency trend
            
            Recommended Actions:
            - Schedule routine maintenance
            - Monitor key performance indicators
            - Plan for preventive maintenance
            - Review maintenance logs
        """)
    else:
        st.success("""
            ✅ System Operating Normally
            
            Status:
            1. System efficiency within target
            2. Few or no anomalies
            3. Stable or positive efficiency trend
            
            Recommended Actions:
            - Continue regular maintenance schedule
            - Monitor system performance
            - Update maintenance logs
            - Plan for future upgrades
        """)
def generate_area_consumption_data():
    """Generate synthetic consumption data for different campus areas"""
    # Base consumption from the CSV data
    base_consumption = {
        'Academic': {
            'Core Academic Complex': 905941,
            'Lecture Hall Complex': 17250,
            'Research Buildings & Labs': 373330,
            'Technology Complex': 140280,
            'Nanotechnology Center': 45920,
            'Central Library': 51000,
            'Central Workshop': 19170
        },
        'Residential': {
            'Residential Quarters': 354780,
            'Other Residences': 92070
        },
        'Utility': {
            'Guest Houses': 120000,
            'IIT Guwahati Hospital': 85000,
            'Power House': 45000,
            'Sewage Treatment Plant': 35000,
            'Student Activity Centers': 60000,
            'Sports Facilities': 40000
        },
        'Commercial': {
            'Major Establishments': 21815,
            'Minor Establishments': 22645
        }
    }
    
    dates = pd.date_range(start='2023-02-01', end='2024-02-05', freq='D')
    data = []
    
    for date in dates:
        # Add variations based on different factors
        for category, buildings in base_consumption.items():
            for building, base_demand in buildings.items():
                # Add daily variations
                time_variation = 1.2 if date.dayofweek < 5 else 0.8  # Less usage on weekends
                
                # Seasonal variations
                month_factor = 1 + 0.2 * np.sin(2 * np.pi * date.month / 12)  # Higher in summer
                
                # Random events
                event_factor = 1.0
                if np.random.random() < 0.05:  # 5% chance of events
                    event_factor = np.random.uniform(0.8, 1.3)
                
                # Calculate actual demand
                demand = base_demand * time_variation * month_factor * event_factor
                
                # Add some random noise
                demand *= np.random.normal(1, 0.05)
                
                data.append({
                    'Date': date,
                    'Category': category,
                    'Building': building,
                    'Demand': demand,
                    'BaseCapacity': base_demand
                })
    
    return pd.DataFrame(data)

def analyze_area_efficiency(df_areas):
    """Analyze efficiency metrics for different areas"""
    latest_date = df_areas['Date'].max()
    current_usage = df_areas[df_areas['Date'] == latest_date].copy()
    
    # Calculate efficiency metrics
    current_usage['Efficiency'] = (current_usage['BaseCapacity'] - current_usage['Demand']) / current_usage['BaseCapacity'] * 100
    current_usage['UtilizationRate'] = (current_usage['Demand'] / current_usage['BaseCapacity']) * 100
    
    # Identify areas of concern
    current_usage['Status'] = np.where(current_usage['UtilizationRate'] > 90, 'Critical',
                               np.where(current_usage['UtilizationRate'] > 80, 'Warning', 'Normal'))
    
    return current_usage

def create_area_analysis_section(df_areas, area_analysis):
    """Create the area analysis section in the dashboard"""
    st.subheader("Campus Areas Water Usage Analysis")
    
    # Area selection
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_category = st.selectbox("Select Category", ['All'] + list(df_areas['Category'].unique()))
    with col2:
        if selected_category == 'All':
            building_options = df_areas['Building'].unique()
        else:
            building_options = df_areas[df_areas['Category'] == selected_category]['Building'].unique()
        selected_building = st.selectbox("Select Building", building_options)
    
    # Detailed area analysis
    building_data = df_areas[df_areas['Building'] == selected_building]
    current_metrics = area_analysis[area_analysis['Building'] == selected_building].iloc[0]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_usage = current_metrics['Demand']
        st.metric("Current Usage", f"{current_usage/1000:.1f} kLD")
    with col2:
        st.metric("Utilization Rate", f"{current_metrics['UtilizationRate']:.1f}%")
    with col3:
        st.metric("Efficiency", f"{current_metrics['Efficiency']:.1f}%")
    with col4:
        base_capacity = current_metrics['BaseCapacity']
        st.metric("Design Capacity", f"{base_capacity/1000:.1f} kLD")
    
    # Usage trends
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        # Building usage trend
        fig = px.line(building_data, x='Date', y='Demand',
                     title=f'{selected_building} Usage Trend')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with row1_col2:
        # Category comparison
        category_usage = area_analysis.groupby('Category')['Demand'].sum().reset_index()
        fig = px.pie(category_usage, values='Demand', names='Category',
                    title='Water Usage Distribution by Category')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Efficiency analysis
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        # Efficiency ranking
        fig = px.bar(area_analysis.sort_values('Efficiency'),
                    x='Building', y='Efficiency',
                    color='Status',
                    title='Building Efficiency Ranking')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with row2_col2:
        # Usage patterns by category
        daily_pattern = building_data.copy()
        daily_pattern['Weekday'] = daily_pattern['Date'].dt.day_name()
        daily_avg = daily_pattern.groupby('Weekday')['Demand'].mean()
        
        fig = px.bar(x=daily_avg.index, y=daily_avg.values,
                    title=f'{selected_building} - Daily Usage Pattern',
                    labels={'x': 'Day of Week', 'y': 'Average Demand (L)'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Status and recommendations
    st.subheader("Status and Recommendations")
    
    utilization = current_metrics['UtilizationRate']
    if utilization > 90:
        st.error(f"""
            🚨 Critical Usage Level for {selected_building}
            
            Current Status:
            - Utilization: {utilization:.1f}% of capacity
            - Category: {current_metrics['Category']}
            
            Recommended Actions:
            1. Immediate usage audit required
            2. Implement water conservation measures
            3. Check for leaks and inefficiencies
            4. Consider capacity expansion
            5. Review peak usage patterns
        """)
    elif utilization > 80:
        st.warning(f"""
            ⚠️ High Usage Level for {selected_building}
            
            Current Status:
            - Utilization: {utilization:.1f}% of capacity
            - Category: {current_metrics['Category']}
            
            Recommended Actions:
            1. Monitor usage patterns closely
            2. Implement basic conservation measures
            3. Schedule maintenance check
            4. Review water-intensive activities
            5. Prepare optimization plan
        """)
    else:
        st.success(f"""
            ✅ Normal Usage Level for {selected_building}
            
            Current Status:
            - Utilization: {utilization:.1f}% of capacity
            - Category: {current_metrics['Category']}
            
            Recommendations:
            1. Maintain current efficiency
            2. Continue regular monitoring
            3. Document best practices
            4. Plan preventive maintenance
            5. Consider additional optimization opportunities
        """)

def main():
    st.set_page_config(page_title="IIT Guwahati Water Management", layout="wide")
    
    # Generate all data
    df_main = generate_synthetic_data()
    df_hostels = generate_hostel_data()
    df_quality = generate_water_quality_data()
    df_areas = generate_area_consumption_data()  # New area data
    
    # Analyze data
    df_anomalies = detect_campus_anomalies(df_main, df_hostels)
    hostel_analysis = analyze_hostel_efficiency(df_hostels)
    quality_analysis, violations = analyze_water_quality(df_quality)
    predictions, confidence_intervals = predict_future_requirements(df_main, df_hostels)
    quality_predictions, quality_confidence_intervals, future_dates = predict_water_quality(df_quality)
    area_analysis = analyze_area_efficiency(df_areas)  # New area analysis
    
    # Dashboard layout
    st.title("IIT Guwahati Water Management Dashboard")
    
    # Enhanced tabs including area analysis
    tabs = st.tabs([
        "Overview",
        "Hostel Analysis",
        "Area Analysis",  # New tab
        "Water Quality",
        "Treatment Optimization",
        "Anomaly Detection",
        "Predictions",
        "Maintenance"
    ])
    
    with tabs[0]:
        create_overview_section(df_main, df_hostels)
    
    with tabs[1]:
        create_hostel_section(df_hostels, hostel_analysis)
    
    with tabs[2]:
        create_area_analysis_section(df_areas, area_analysis)  # New section
    
    with tabs[3]:
        create_water_quality_section(
            df_quality,
            quality_analysis,
            violations,
            quality_predictions,
            quality_confidence_intervals,
            future_dates
        )
    
    with tabs[4]:
        create_treatment_optimization_section(df_quality)
    
    with tabs[5]:
        create_anomaly_section(df_anomalies)
    
    with tabs[6]:
        create_prediction_section(df_main, df_hostels, predictions, confidence_intervals)
    
    with tabs[7]:
        create_maintenance_section(df_main, df_anomalies)

if __name__ == "__main__":
    main()

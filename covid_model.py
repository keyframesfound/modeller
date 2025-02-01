import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime
import seaborn as sns

def logistic_function(x, L, k, x0):
    """
    L: maximum limit (carrying capacity)
    k: steepness of the curve
    x0: x-value of the sigmoid's midpoint
    """
    return L / (1 + np.exp(-k * (x - x0)))

def analyze_covid_data(data_file):
    try:
        # Read and validate data
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
        df['days'] = (df['date'] - df['date'].min()).dt.days

        # Set style for better visualization
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Cumulative Cases with Model Fit
        ax1 = plt.subplot(211)
        
        # Normalize data for better fitting
        y_data = df['cases'].values
        x_data = df['days'].values
        
        # Initial parameter estimates
        p0 = [
            df['cases'].max() * 1.2,  # L: slightly higher than max observed
            0.1,                      # k: initial growth rate guess
            np.median(x_data)         # x0: middle of time series
        ]
        
        # Fit logistic model with bounds
        bounds = (
            [0, 0.01, 0],            # lower bounds
            [np.inf, 1, np.inf]      # upper bounds
        )
        popt, pcov = curve_fit(logistic_function, x_data, y_data,
                              p0=p0,
                              bounds=bounds,
                              maxfev=10000)
        
        # Calculate 95% confidence intervals
        perr = np.sqrt(np.diag(pcov))
        
        # Generate prediction curve
        x_fit = np.linspace(0, len(df) + 30, 100)
        y_fit = logistic_function(x_fit, *popt)
        
        # Plot actual data and fit
        ax1.scatter(df['date'], df['cases'], label='Actual Data', alpha=0.6)
        ax1.plot(pd.date_range(start=df['date'].min(), 
                              periods=len(x_fit), 
                              freq='D'), 
                y_fit, 'r-', 
                label='Logistic Model', 
                linewidth=2)
        
        # Add formula text
        formula = f'$f(t) = \\frac{{{popt[0]:,.0f}}}{{1 + e^{{-{popt[1]:.3f}(t - {popt[2]:.1f})}}}}$'
        ax1.text(0.05, 0.95, formula,
                transform=ax1.transAxes, fontsize=12, verticalalignment='top')
        
        ax1.set_ylabel('Cumulative Cases')
        ax1.set_title('COVID-19 Cases: Actual vs Model Prediction')
        ax1.legend()
        ax1.grid(True)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 2: Daily New Cases
        ax2 = plt.subplot(212)
        df['daily_cases'] = df['cases'].diff()
        ax2.bar(df['date'], df['daily_cases'], alpha=0.6, label='Daily New Cases')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Daily New Cases')
        ax2.set_title('Daily New COVID-19 Cases')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Formatting
        plt.tight_layout()
        
        # Save high-resolution plot
        plt.savefig('covid_analysis.png', dpi=300, bbox_inches='tight')
        
        # Display plot if running in interactive mode
        plt.show()
        
        # Calculate model parameters
        L, k, x0 = popt
        results = {
            'carrying_capacity': L,
            'growth_rate': k,
            'midpoint_day': x0,
            'r_squared': calculate_r_squared(df['cases'], 
                                          logistic_function(df['days'], *popt)),
            'total_cases': df['cases'].iloc[-1],
            'peak_daily_cases': df['daily_cases'].max(),
            'data_points': len(df)
        }
        
        # Print summary statistics
        print("\nCOVID-19 Analysis Summary:")
        print("-" * 30)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:,.2f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value:,}")
        
        return results
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return None

def calculate_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

if __name__ == "__main__":
    results = analyze_covid_data('covid_data.csv')
    if results:
        print("\nModel Results:")
        print("--------------")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")

import pandas as pd
import requests
from datetime import datetime

def download_covid_data():
    # Correct URL for JHU CSSE COVID-19 global confirmed cases
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    
    try:
        # Read the time series data
        df = pd.read_csv(url)
        
        # Melt the dataframe to convert to long format
        df_melted = df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
                           var_name='date', value_name='cases')
        
        # Convert date strings to datetime
        df_melted['date'] = pd.to_datetime(df_melted['date'])
        
        # Group by date to get global total
        global_cases = df_melted.groupby('date')['cases'].sum().reset_index()
        
        # Sort by date
        global_cases = global_cases.sort_values('date')
        
        # Convert date to desired format
        global_cases['date'] = global_cases['date'].dt.strftime('%Y-%m-%d')
        
        # Save to CSV
        global_cases.to_csv('covid_data_complete.csv', index=False)
        print(f"Data downloaded successfully. Total days: {len(global_cases)}")
        
    except Exception as e:
        print(f"Error downloading data: {str(e)}")

if __name__ == "__main__":
    download_covid_data()

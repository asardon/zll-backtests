import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import numpy as np

def compute_loss_and_default(start_index, price_column_name, loan_tenor, initial_loan_value, df):
    if start_index + loan_tenor >= len(df):
        return None, None
    end_price = df.iloc[start_index + loan_tenor][price_column_name]
    defaulted = end_price < initial_loan_value
    loss = (1 - end_price / initial_loan_value) if defaulted else 0
    return defaulted, loss

def simulate_default_and_loss_rate(merged_df, price_column_name, loan_tenors, ltv_ratios):
    results = []
    for i, row in merged_df.iterrows():
        for loan_tenor in loan_tenors:
            for ltv_ratio in ltv_ratios:
                loan_value = row[price_column_name] * ltv_ratio
                defaulted, loss = compute_loss_and_default(i, price_column_name, loan_tenor, loan_value, merged_df)
                if defaulted is not None:
                    results.append({
                        'start_date': row['snapped_at'],
                        'loan_tenor': loan_tenor,
                        'ltv_ratio': ltv_ratio,
                        'defaulted': defaulted,
                        'loss': loss
                    })

    df = pd.DataFrame(results)
    aggregated = df.groupby(['ltv_ratio', 'loan_tenor']).agg(
        start_date=('start_date', 'first'),
        default_rate=('defaulted', 'mean'),
        average_loss=('loss', 'mean'),
        percentile_loss_90=('loss', lambda x: x.quantile(0.90)),
        percentile_loss_95=('loss', lambda x: x.quantile(0.95)),
        percentile_loss_99=('loss', lambda x: x.quantile(0.99)),
        percentile_loss_worst=('loss', lambda x: x.max()),
        observation_count=('defaulted', 'size'),
        loss_stddev=('loss', 'std')
    ).reset_index()
    
    # Prepare distribution data
    distribution_data = []
    for loan_tenor in loan_tenors:
        for ltv_ratio in ltv_ratios:
            subset = df[(df['loan_tenor'] == loan_tenor) & (df['ltv_ratio'] == ltv_ratio)]
            distribution_data.append({
                'loan_tenor': loan_tenor,
                'ltv_ratio': ltv_ratio,
                'loss_values': subset['loss'].tolist()
            })
            
    return aggregated, df, distribution_data

def get_available_currencies():
    csv_files = [file for file in os.listdir() if file.endswith('.csv')]
    return [file.split('.')[0].upper() for file in csv_files]

def read_and_process_data(collateral_currency, loan_currency):
    if loan_currency == 'USD':
        df = pd.read_csv(collateral_currency.lower() + '.csv', sep=",")
        df['price'] = df['price']  # Explicit assignment, may be unnecessary
    elif collateral_currency == 'USD':
        df = pd.read_csv(loan_currency.lower() + '.csv', sep=",")
        df['price'] = 1 / df['price']  # Reciprocal
    else:
        df1 = pd.read_csv(collateral_currency.lower() + '.csv', sep=",")
        df2 = pd.read_csv(loan_currency.lower() + '.csv', sep=",")
        df = df1.merge(df2, on='snapped_at', suffixes=('_coll', '_loan'))
        df['price'] = df['price_coll'] / df['price_loan']  # Cross price
    df.sort_values(by='snapped_at', ascending=True, inplace=True)
    return df

def compute_trailing_volatility(price_series, window=30):
    """Compute the trailing volatility of returns for a given price series."""
    returns = price_series.pct_change().dropna()
    volatility = returns.rolling(window).std() * (252 ** 0.5)  # Annualized volatility
    return volatility

def plot_price_over_time(df, selected_from_date, selected_to_date, collateral_currency, loan_currency):
    df_filtered = df[(df['snapped_at'] >= str(selected_from_date)) & (df['snapped_at'] <= str(selected_to_date))]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    ax1.fill_between(pd.to_datetime(df_filtered['snapped_at']), df_filtered['price'], color="skyblue", label='Price', alpha=0.5)
    ax1.plot(pd.to_datetime(df_filtered['snapped_at']), df_filtered['price'], color='blue')
    ax1.set_title('Price Over Time: {} / {}'.format(collateral_currency, loan_currency), fontsize=22, fontweight="bold")
    ax1.set_ylabel('{} Price (in {})'.format(collateral_currency, loan_currency), fontsize=20, fontweight="bold")
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Annotations and lines for high, low, and latest price on ax1 (Price plot)
    highest_price = df_filtered['price'].max()
    lowest_price = df_filtered['price'].min()
    last_price = df_filtered['price'].iloc[-1]

    # Check proximity for Price annotations
    price_threshold = 0.05 * (highest_price - lowest_price)
    annotate_high_price = True
    annotate_low_price = True

    if abs(highest_price - last_price) < price_threshold:
        annotate_high_price = False
    if abs(lowest_price - last_price) < price_threshold:
        annotate_low_price = False

    # Add Price annotations based on proximity
    if annotate_high_price:
        ax1.axhline(highest_price, color='green', linestyle='--', xmax=0.98)
        ax1.annotate(f"Highest Price: {highest_price:.2f} {loan_currency}", 
                     (1.01, highest_price), 
                     xycoords=("axes fraction", "data"),
                     textcoords="offset points",
                     xytext=(15, 0),
                     fontsize=15,
                     color='green')
    if annotate_low_price:
        ax1.axhline(lowest_price, color='red', linestyle='--', xmax=0.98)
        ax1.annotate(f"Lowest Price: {lowest_price:.2f} {loan_currency}", 
                     (1.01, lowest_price), 
                     xycoords=("axes fraction", "data"),
                     textcoords="offset points",
                     xytext=(15, 0),
                     fontsize=15,
                     color='red')
    ax1.axhline(last_price, color='blue', linestyle='--', xmax=0.98)
    ax1.annotate(f"Latest Price: {last_price:.2f} {loan_currency}", 
                 (1.01, last_price), 
                 xycoords=("axes fraction", "data"),
                 textcoords="offset points",
                 xytext=(15, 0),
                 fontsize=15,
                 color='blue')
    
    # Calculate trailing volatility (e.g., 30-day rolling std deviation)
    df_filtered['log_returns'] = np.log(df_filtered['price'] / df_filtered['price'].shift(1))
    df_filtered['volatility'] = df_filtered['log_returns'].rolling(window=30).std() * (365**0.5)

    ax2.plot(pd.to_datetime(df_filtered['snapped_at']), df_filtered['volatility']*100, color='red', label='Volatility')
    ax2.fill_between(pd.to_datetime(df_filtered['snapped_at']), df_filtered['volatility']*100, color="red", alpha=0.3)
    ax2.set_title('Annualized Volatility: {} / {} \n (Trailing 30-day)'.format(collateral_currency, loan_currency), fontsize=20, fontweight="bold")
    ax2.set_ylabel('Volatility (in %)', fontsize=18, fontweight="bold")
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Annotations and lines for high, low, and latest volatility on ax2 (Volatility plot)
    highest_volatility = df_filtered['volatility'].max()*100
    lowest_volatility = df_filtered['volatility'].min()*100
    last_volatility = df_filtered['volatility'].iloc[-1]*100

    # Check proximity for Volatility annotations
    volatility_threshold = 0.05 * (highest_volatility - lowest_volatility)
    annotate_high_volatility = True
    annotate_low_volatility = True

    if abs(highest_volatility - last_volatility) < volatility_threshold:
        annotate_high_volatility = False
    if abs(lowest_volatility - last_volatility) < volatility_threshold:
        annotate_low_volatility = False

    # Add Volatility annotations based on proximity
    if annotate_high_volatility:
        ax2.axhline(highest_volatility, color='green', linestyle='--', xmax=0.98)
        ax2.annotate(f"Highest Vol.: {highest_volatility:.2f}% p.a.", 
                     (1.01, highest_volatility), 
                     xycoords=("axes fraction", "data"),
                     textcoords="offset points",
                     xytext=(15, 0),
                     fontsize=15,
                     color='green')
    if annotate_low_volatility:
        ax2.axhline(lowest_volatility, color='red', linestyle='--', xmax=0.98)
        ax2.annotate(f"Lowest Vol.: {lowest_volatility:.2f}% p.a.", 
                     (1.01, lowest_volatility), 
                     xycoords=("axes fraction", "data"),
                     textcoords="offset points",
                     xytext=(15, 0),
                     fontsize=15,
                     color='red')
    ax2.axhline(last_volatility, color='blue', linestyle='--', xmax=0.98)
    ax2.annotate(f"Latest Vol.: {last_volatility:.2f}% p.a.", 
                 (1.01, last_volatility), 
                 xycoords=("axes fraction", "data"),
                 textcoords="offset points",
                 xytext=(15, 0),
                 fontsize=15,
                 color='blue')
    
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.tick_params(axis='both', which='major', labelsize=14)

    fig.tight_layout()
    fig.autofmt_xdate()
    
    return fig

def plot_heatmap(aggregated, value_column, title, cmap='coolwarm', save_as=None):
    aggregated['ltv_ratio'] = (aggregated['ltv_ratio'] * 100).astype(int)
    
    # Create a pivot table for the heatmap
    heatmap_data = aggregated.pivot_table(index="ltv_ratio", columns="loan_tenor", values=value_column, aggfunc='mean') * 100
    
    # Calculate annotations with a % suffix
    annotations = heatmap_data.applymap(lambda x: f"{int(x)}%")
    
    plt.figure(figsize=(20, 15))
    ax = sns.heatmap(
        heatmap_data, 
        annot=annotations.values,  # Use the custom annotations
        cmap=cmap, 
        cbar_kws={'label': title + ' (%)'}, 
        fmt='',  # No special format for annotations since they're already strings
        annot_kws={"size": 16, "weight": "bold"},  # Increase annotation font size and make it bold for readability
        linewidths=.5,  # Add lines between cells for clarity
        center=heatmap_data.mean().mean(),  # Center the colormap around the mean value
    )
    plt.title(title, fontsize=20)
    plt.show()

    # Update cbar (colorbar) label and ticks font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(title + ' (%)', size=18)

    plt.title("{}".format(title), fontsize=20, fontweight="bold")
    plt.xlabel("Loan Tenor (in days)", fontsize=18, fontweight="bold")
    plt.ylabel("LTV Ratio (in %)", fontsize=18, fontweight="bold")
    ax.invert_yaxis()  # To display the highest LTV ratio at the top
    ax.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label size
    if save_as:
        plt.savefig(save_as)
    return plt

def main():
    st.title("Backtesting Period")
    
    # Sidebar for input selection
    with st.sidebar:
        currencies = ['USD'] + get_available_currencies()
        collateral_currency = st.selectbox("Select Collateral Currency", currencies, index=currencies.index('RPL'))
        
        # Filter out the chosen collateral currency for loan selection
        loan_currencies = [currency for currency in currencies if currency != collateral_currency]
        loan_currency = st.selectbox("Select Loan Currency", loan_currencies, index=currencies.index('USD'))

        df = read_and_process_data(collateral_currency, loan_currency)

        # Set the min and max date from the dataframe df
        min_date = pd.to_datetime(df['snapped_at'].min())
        max_date = pd.to_datetime(df['snapped_at'].max())

        selected_from_date, selected_to_date = st.date_input("Select Date Range", [min_date, max_date])
        loan_tenors = st.multiselect("Select Tenors", [1,2,3,4,5,6,7,10,20,30,60,90,120,150,180,365], default=[30,60,90])
        selected_ltv_ratios = st.multiselect("Select LTV Ratios", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99], default=[0.3, 0.4, 0.5])
        
        metric = st.selectbox(
            "Select Metric to Display",
            ["VaR (95th Percentile Loss)", "VaR (90th Percentile Loss)", "VaR (99th Percentile Loss)", "VaR (Worst Case Loss)", "Average Loss", "Default Rate"]
        )
        
        metrics_map = {
            "VaR (90th Percentile Loss)": ("percentile_loss_90", "VaR (90th Percentile Loss)"),
            "VaR (95th Percentile Loss)": ("percentile_loss_95", "VaR (95th Percentile Loss)"),
            "VaR (99th Percentile Loss)": ("percentile_loss_99", "VaR (99th Percentile Loss)"),
            "VaR (Worst Case Loss)": ("percentile_loss_worst", "VaR (Worst Case Loss)"),
            "Average Loss": ("average_loss", "Average Loss"),
            "Default Rate": ("default_rate", "Default Rate")
        }
    
    # Read and process data
    df = read_and_process_data(collateral_currency, loan_currency)
        
    # Plot the time series at the top
    time_series_fig = plot_price_over_time(df, selected_from_date, selected_to_date, collateral_currency, loan_currency)
    st.pyplot(time_series_fig)

    column_name, title = metrics_map[metric]
    
    aggregated, _, _ = simulate_default_and_loss_rate(df, 'price', loan_tenors, selected_ltv_ratios)

    st.title(f"Lender Risk Analysis")

    fig = plot_heatmap(aggregated, column_name, title, "Reds", None)
    st.write(f"Heatmap illustrates your {title} across different tenor and LTV combinations. The {title} reflect a scenario where you randomly loaned {loan_currency} against {collateral_currency} collateral during the backtest period, with 0% APR and no upfront fees.")
    st.pyplot(fig)  # Display the heatmap plot
    
if __name__ == "__main__":
    main()
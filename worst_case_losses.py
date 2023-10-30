import os

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

def compute_loss_and_default(start_index, price_column_name, loan_tenor, initial_loan_value, upfront_fee, APR, df):
    if start_index + loan_tenor >= len(df):
        return None, None, None, None, None, None, None
    end_price = df.iloc[start_index + loan_tenor][price_column_name]
    
    start_price = df.iloc[start_index][price_column_name]

    repayment_amount = initial_loan_value * (1 + APR * loan_tenor / 365)
    upfront_fee_amount = upfront_fee * start_price
    
    defaulted = end_price * (1 - upfront_fee) < repayment_amount
    loss = (1 - end_price / initial_loan_value) if end_price < initial_loan_value else 0
    apy_from_interest_given_no_default = 0 if defaulted else APR
    apy_from_upfront_given_no_default = upfront_fee_amount / initial_loan_value * 365 / loan_tenor

    roi_from_interest = 0 if defaulted else (repayment_amount - initial_loan_value)/initial_loan_value
    roi_from_upfront_fee = upfront_fee_amount / initial_loan_value
    return defaulted, loss, apy_from_interest_given_no_default, apy_from_upfront_given_no_default, end_price, roi_from_interest, roi_from_upfront_fee

def simulate_specific_strategy(df, price_column_name, loan_tenor, ltv_ratio, upfront_fee, APR):
    results = []
    df = df.reset_index()
    for i, row in df.iterrows():
        loan_value = row[price_column_name] * ltv_ratio
        defaulted, loss, apy_from_interest_given_no_default, apy_from_upfront_given_no_default, end_price, roi_from_interest, roi_from_upfront_fee = compute_loss_and_default(i, price_column_name, loan_tenor, loan_value, upfront_fee, APR, df)
        if defaulted is not None:
            results.append({
                'price': row[price_column_name],
                'price_at_expiry': end_price,
                'loan_inception': row['snapped_at_datetime'],
                'loan_expiry': row['snapped_at_datetime'] + pd.Timedelta(days=loan_tenor),
                'defaulted': defaulted,
                'neg_loss': -loss,
                'apy_from_interest_given_no_default': apy_from_interest_given_no_default,
                'apy_from_upfront_given_no_default': apy_from_upfront_given_no_default,
                'pnl_total': apy_from_interest_given_no_default + apy_from_upfront_given_no_default - loss,
                'roi_from_interest': roi_from_interest,
                'roi_from_upfront_fee': roi_from_upfront_fee
            })
    return results
                
def simulate_default_and_loss_rate(merged_df, price_column_name, loan_tenors, ltv_ratios, upfront_fee, APR):
    results = []
    for i, row in merged_df.iterrows():
        for loan_tenor in loan_tenors:
            for ltv_ratio in ltv_ratios:
                loan_value = row[price_column_name] * ltv_ratio
                defaulted, loss, apy_from_interest_given_no_default, apy_from_upfront_given_no_default, _, _, _ = compute_loss_and_default(i, price_column_name, loan_tenor, loan_value, upfront_fee, APR, merged_df)
                if defaulted is not None:
                    results.append({
                        'start_date': row['snapped_at_datetime'],
                        'loan_tenor': loan_tenor,
                        'ltv_ratio': ltv_ratio,
                        'defaulted': defaulted,
                        'loss': loss,
                        'apy_from_interest_given_no_default': apy_from_interest_given_no_default,
                        'apy_from_upfront_given_no_default': apy_from_upfront_given_no_default,
                        'pnl_total': apy_from_interest_given_no_default + apy_from_upfront_given_no_default - loss
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
        pnl_mean=('pnl_total', 'mean'),
        pnl_median=('pnl_total', 'median')
    ).reset_index()
            
    # Calculate PnL percentiles for each LTV and tenor combination
    for ltv_ratio in ltv_ratios:
        for loan_tenor in loan_tenors:
            sub_df = df[(df['ltv_ratio'] == ltv_ratio) & (df['loan_tenor'] == loan_tenor)]
            for p in np.arange(0, 1.05, 0.05):
                percentile_pnl = sub_df['pnl_total'].quantile(p)
                aggregated.loc[(aggregated['ltv_ratio'] == ltv_ratio) & (aggregated['loan_tenor'] == loan_tenor), f'percentile_pnl_{int(p*100)}'] = percentile_pnl
    return aggregated, df

def get_available_currencies():
    csv_files = [file for file in os.listdir() if file.endswith('.csv')]
    return [file.split('.')[0].upper() for file in csv_files]

def read_and_process_data(collateral_currency, loan_currency):
    if loan_currency == 'USD':
        df = pd.read_csv(collateral_currency.lower() + '.csv', sep=",")
        df['price'] = df['price']  # Explicit assignment, may be unnecessary
        df['snapped_at_datetime'] = pd.to_datetime(df['snapped_at'])
    elif collateral_currency == 'USD':
        df = pd.read_csv(loan_currency.lower() + '.csv', sep=",")
        df['price'] = 1 / df['price']  # Reciprocal
        df['snapped_at_datetime'] = pd.to_datetime(df['snapped_at'])
    else:
        df1 = pd.read_csv(collateral_currency.lower() + '.csv', sep=",")
        df2 = pd.read_csv(loan_currency.lower() + '.csv', sep=",")
        df1['snapped_at_datetime'] = pd.to_datetime(df1['snapped_at'])
        df2['snapped_at_datetime'] = pd.to_datetime(df2['snapped_at'])
        df = df1.merge(df2, on='snapped_at_datetime', suffixes=('_coll', '_loan'))
        df['price'] = df['price_coll'] / df['price_loan']  # Cross price
    df.sort_values(by='snapped_at_datetime', ascending=True, inplace=True)
    return df

def compute_trailing_volatility(price_series, window=30):
    """Compute the trailing volatility of returns for a given price series."""
    returns = price_series.pct_change().dropna()
    volatility = returns.rolling(window).std() * (252 ** 0.5)  # Annualized volatility
    return volatility

def format_two_significant_digits(num, _=None):
    if num == 0:
        return "0.00"
    magnitude = int(np.floor(np.log10(abs(num))))
    if magnitude >= 1:
        return f"{num:.2f}"
    else:
        format_str = "{:." + str(2 - magnitude) + "f}"
        return format_str.format(num)

def plot_price_over_time(df, selected_from_date, selected_to_date, collateral_currency, loan_currency):
    df_filtered = df[(df['snapped_at_datetime'] >= pd.to_datetime(selected_from_date).tz_localize('UTC')) & (df['snapped_at_datetime'] <= pd.to_datetime(selected_to_date).tz_localize('UTC'))]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    ax1.fill_between(pd.to_datetime(df_filtered['snapped_at_datetime']), df_filtered['price'], color="skyblue", label='Price', alpha=0.5)
    ax1.plot(pd.to_datetime(df_filtered['snapped_at_datetime']), df_filtered['price'], color='blue')
    ax1.set_title('Price Over Time: {} / {}'.format(collateral_currency, loan_currency), fontsize=22, fontweight="bold")
    ax1.set_ylabel('{} Price (in {})'.format(collateral_currency, loan_currency), fontsize=20, fontweight="bold")
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_two_significant_digits))
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
        ax1.annotate(f"Highest Price: {format_two_significant_digits(highest_price)} {loan_currency}", 
             (1.01, highest_price), 
             xycoords=("axes fraction", "data"),
             textcoords="offset points",
             xytext=(15, 0),
             fontsize=15,
             color='green')
    if annotate_low_price:
        ax1.axhline(lowest_price, color='red', linestyle='--', xmax=0.98)
        ax1.annotate(f"Lowest Price: {format_two_significant_digits(lowest_price)} {loan_currency}",
                     (1.01, lowest_price), 
                     xycoords=("axes fraction", "data"),
                     textcoords="offset points",
                     xytext=(15, 0),
                     fontsize=15,
                     color='red')
    ax1.axhline(last_price, color='blue', linestyle='--', xmax=0.98)
    ax1.annotate(f"Latest Price: {format_two_significant_digits(last_price)} {loan_currency}",
                 (1.01, last_price), 
                 xycoords=("axes fraction", "data"),
                 textcoords="offset points",
                 xytext=(15, 0),
                 fontsize=15,
                 color='blue')
    
    # Calculate trailing volatility (e.g., 30-day rolling std deviation)
    df_filtered['log_returns'] = np.log(df_filtered['price'] / df_filtered['price'].shift(1))
    df_filtered['volatility'] = df_filtered['log_returns'].rolling(window=30).std() * (365**0.5)

    ax2.plot(df_filtered['snapped_at_datetime'], df_filtered['volatility']*100, color='red', label='Volatility')
    ax2.fill_between(df_filtered['snapped_at_datetime'], df_filtered['volatility']*100, color="red", alpha=0.3)
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
    
    return fig, df_filtered

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
        min_date = df['snapped_at_datetime'].min()
        max_date = df['snapped_at_datetime'].max()

        selected_from_date, selected_to_date = st.date_input("Select Date Range", [min_date, max_date])
        loan_tenors = st.multiselect("Select Tenors", [1,2,3,4,5,6,7,10,20,30,60,90,120,150,180,365], default=[30,60,90])
        selected_ltv_ratios = st.multiselect("Select LTV Ratios", np.around(np.append(np.arange(0.05, 1, 0.05), 0.99), 2), default=[0.3, 0.4, 0.5])
        
        APR = st.number_input("Enter Annual Percentage Rate (APR) in percentage:", min_value=0.0, max_value=500.0, value=5.0)
        upfront_fee = st.number_input("Enter Upfront Fee in percentage:", min_value=0.0, max_value=100.0, value=0.0)
        
        # Convert percentage to proportions for calculations
        APR /= 100.0
        upfront_fee /= 100.0

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
    time_series_fig, df_filtered = plot_price_over_time(df, selected_from_date, selected_to_date, collateral_currency, loan_currency)
    st.pyplot(time_series_fig)

    column_name, title = metrics_map[metric]
    
    aggregated, _ = simulate_default_and_loss_rate(df, 'price', loan_tenors, selected_ltv_ratios, upfront_fee, APR)

    st.title(f"Risk Analysis")

    fig = plot_heatmap(aggregated, column_name, title, "Reds", None)
    st.write(f"Heatmap illustrates your {title} across different tenor and LTV combinations. The {title} figures below are for a scenario where you would've randomly loaned {loan_currency} against {collateral_currency} collateral during the backtest period and borrower default would've ocurred if the collateral price would've been worth less then the debt owed.")
    st.pyplot(fig)

    

    st.title(f"Detailed Analysis")
    st.write(f"For a more detailed analysis you can input your desired LTV, loan tenor, APR, and upfront fee you'd like to backtest.")
    ltv_detailed = st.number_input("LTV:", min_value=0.01, max_value=1.0, value=.2, format="%.4f")
    tenor_detailed = st.number_input("Loan Tenor (days):", min_value=1, max_value=365, value=30)
    apr_detailed = st.number_input("APR:", min_value=0.0, max_value=1.0, value=0.05, format="%.4f")
    upfront_fee_detailed = st.number_input("Upfront Fee:", min_value=0.0, max_value=100.0, value=0.0, format="%.4f")
    st.write(f"These loan parameter values are applied across the entire backtesting period, assuming that each day a borrower would have secured a loan from you under these terms. The resulting loan-per-collateral unit over time is displayed, pinpointing specific moments and loan-per-collateral values where a default might have occurred. Additionally, the subsequent chart illustrates your RoI over time: losses are emphasized in red, while gains from interest and upfront fees are depicted in green. We also present the cumulative RoI, representing the returns you would have amassed if you continuously extended loans under these conditions, with borrowers consistently seeking loans from you on a daily basis.")

    results_detailed = simulate_specific_strategy(df_filtered, 'price', tenor_detailed, ltv_detailed, upfront_fee_detailed, apr_detailed)
    results_df = pd.DataFrame(results_detailed)
    # Adjust figure size and create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14,14), gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.suptitle('Price and Loan-per-Collateral Unit over Time', fontsize=18, fontweight='bold')

    # --- AX1: Price and LTV*Price Over Time ---
    ax1.plot(results_df['loan_inception'], results_df['price'], label='Price at loan origination', color='black', linewidth=1.5)
    ax1.plot(results_df['loan_inception'], results_df['price_at_expiry'], label='Price at loan expiry', color='blue', alpha=0.7, linewidth=1, linestyle=':')
    ax1.plot(results_df['loan_inception'], results_df['price']*ltv_detailed, label='Loan per Collateral', linestyle=':', color='gray', linewidth=1)

    defaulted_dates = results_df.loc[results_df['defaulted'], 'loan_inception']
    defaulted_values = (results_df.loc[results_df['defaulted'], 'price'] * ltv_detailed).values
    ax1.scatter(defaulted_dates, defaulted_values, color='red', marker='x', label='Defaults', zorder=5)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_two_significant_digits))
    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel('{} Price (in {})'.format(collateral_currency, loan_currency), fontsize=20, fontweight="bold")
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey', alpha=0.5)
    ax1.legend(fontsize=12, loc='upper left')

    # Get ROI data
    results_df['roi_from_interest_and_upfront_fee'] = results_df['roi_from_interest'] + results_df['roi_from_upfront_fee']
    
    # Plot Earnings as area
    ax2.fill_between(results_df['loan_inception'], results_df['roi_from_interest_and_upfront_fee'], color='lightgreen', label='Earnings', where=(results_df['roi_from_interest_and_upfront_fee'] > 0))

    # Plot Losses as area
    ax2.fill_between(results_df['loan_inception'], results_df['neg_loss'], color='lightcoral', label='Losses', where=(results_df['neg_loss'] < 0))

    # Formatting ax2
    ax2.set_xlabel('Date', fontsize=14)
    ax2.set_ylabel('RoI (%)', fontsize=14)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey', alpha=0.5, axis='y')
    ax2.legend(fontsize=12)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1))

    # --- AX3: Cumulative RoI Over Time ---
    roi_df = pd.DataFrame({
        'loan_inception': results_df['loan_inception'],
        'roi': results_df['roi_from_interest'] + results_df['roi_from_upfront_fee'] + results_df['neg_loss']
    })

    roi_df.dropna(inplace=True)
    roi_df.sort_values(by='loan_inception', inplace=True)
    roi_df['cumulative_roi'] = roi_df['roi'].cumsum()

    ax3.plot(roi_df['loan_inception'], roi_df['cumulative_roi'], color='purple', label='Cumulative RoI', linewidth=1.5)
    ax3.fill_between(roi_df['loan_inception'], 0, roi_df['cumulative_roi'], where=roi_df['cumulative_roi'] >= 0, facecolor='lightgreen', interpolate=True)
    ax3.fill_between(roi_df['loan_inception'], 0, roi_df['cumulative_roi'], where=roi_df['cumulative_roi'] < 0, facecolor='lightcoral', interpolate=True)
    ax3.set_xlabel('Date', fontsize=14)
    ax3.yaxis.set_major_formatter(ticker.PercentFormatter(1))  # Setting y-axis to percentage format
    ax3.set_ylabel('Cumulative RoI', fontsize=14)
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey', alpha=0.5)
    ax3.legend(fontsize=12, loc='upper left')

    # Compute total duration
    total_days = (results_df['loan_inception'].max() - results_df['loan_inception'].min()).days

    for ax in [ax1, ax2, ax3]:
        # Based on total duration, set tick frequency
        if total_days <= 30:  # Less than a month
            locator = mdates.DayLocator()
            formatter = mdates.DateFormatter('%Y-%m-%d')
            
        elif total_days <= 180:  # Less than half a year
            locator = mdates.MonthLocator()
            formatter = mdates.DateFormatter('%Y-%m')
            
        elif total_days <= 365:  # Less than a year
            locator = mdates.MonthLocator(bymonth=[1,3,6,9])
            formatter = mdates.DateFormatter('%Y-%m')   
        else:
            locator = mdates.MonthLocator(bymonth=[1,6])
            formatter = mdates.DateFormatter('%Y-%m')

        # Set the determined locator and formatter to the axis
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    fig.autofmt_xdate()

    plt.tight_layout()
    st.pyplot(fig)

    # Calculate the ROI data
    roi_values = results_df['roi_from_interest'] + results_df['roi_from_upfront_fee'] + results_df['neg_loss']

    # Extract the required percentiles
    percentiles = [10, 5, 1]
    percentile_values = np.percentile(results_df['neg_loss'], percentiles) # Note: Taking negative to represent losses

    # Worst case loss
    worst_case_loss = results_df['neg_loss'].min()
    exp_loss_given_default = results_df['neg_loss'].mean()

    # Plot the histogram
    plt.figure(figsize=(12, 6))
    n, bins, patches = plt.hist(roi_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7, range=(max(-1, worst_case_loss*1.1), roi_values.max()), label='RoI Distribution')
    plt.xlabel('RoI (%)')
    plt.ylabel('Frequency')
    plt.title('Loss and RoI Distribution')

    # Add vertical lines for the percentiles and worst case
    colors = ['green', 'orange', 'purple', 'red']
    labels = [f"VaR ({100 - p}%)" for p in percentiles] + ["Worst Case Loss"]
    lines = []

    line = plt.axvline(x=exp_loss_given_default, color='lightcoral', linestyle='--', linewidth=1.5, label=f'Exp. loss given default: {exp_loss_given_default*100:.2f}%')
    lines.append(line)
        
    for p_val, color, label in zip(percentile_values.tolist() + [worst_case_loss], colors, labels):
        line = plt.axvline(x=p_val, color=color, linestyle='--', linewidth=1.5, label=f'{label}: {p_val*100:.2f}%')
        lines.append(line)
        

    # Legend, including for the histogram
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, loc='upper right', fontsize=10)

    # Format x-axis as percentage
    plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(1))

    # Scale x-axis
    plt.xlim(max(-1, worst_case_loss*1.1), roi_values.max())

    plt.tight_layout()

    st.write(
    """
    The histogram below visaulizes the RoI distribution as well highlights important loss metrics:
    - **Value at Risk (VaR) Percentiles:** Vertical lines indicating potential losses at different confidence levels.
    - **Expected Loss Given Default:** Average loss if a borrower defaults.
    - **Worst Case Loss:** The most significant loss observed.
    """
    )
    st.pyplot(plt)

if __name__ == "__main__":
    main()
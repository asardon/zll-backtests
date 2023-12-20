import os

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from scipy.stats import norm
import requests
from datetime import datetime, timedelta
import base64

# Function to generate a download link for the CSV
def generate_csv_download_link(df, filename="results.csv", text="Download CSV"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

@st.cache_data
def get_available_tokens():
    try:
        # CoinGecko API endpoint to get a list of cryptocurrencies
        api_url = "https://api.coingecko.com/api/v3/coins/list"

        # Send a GET request to the API
        response = requests.get(api_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # Extract and return the list of token symbols
            token_dropdown_names = []
            token_dropdown_names_to_id = {}
            for entry in data:
                dropdown_name = f'{entry["name"]} ({entry["symbol"].upper()})'
                token_dropdown_names.append(dropdown_name)
                token_dropdown_names_to_id[dropdown_name] = {"id": entry["id"], "name": entry["name"], "symbol": entry["symbol"].upper()}
            return token_dropdown_names, token_dropdown_names_to_id

        else:
            # Handle the case when the request was not successful
            print("Error: Unable to fetch token list from CoinGecko API.")
            st.error(f"Error: Unable to fetch token names. Status code: {response.status_code}")
            return None, None

    except Exception as e:
        print("Error:", str(e))
        st.error(f"Error: Unable to fetch token names. Status code: {response.status_code}")
        return None, None

@st.cache_data
def get_historical_data_cg(token_id, vs_currency, date_from, date_to):
    if date_from >= date_to:
        st.warning("Start date should be earlier than end date.")
        return None

    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart/range"

    params = {
        'vs_currency': vs_currency,
        'from': int(date_from.timestamp()),
        'to': int(date_to.timestamp())
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
            return prices
        else:
            st.error(f"Error: Unable to fetch historical price data for token {token_id}. Status code: {response.status_code} {params}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def black_scholes_call(S, K, t, r, sigma):
    t = t / 365.

    # Calculate d1 and d2 parameters
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    # Calculate call option price
    if t > 0:
        delta = norm.cdf(d1)
        call_price = (S * delta - K * np.exp(-r * t) * norm.cdf(d2))
    else:
        delta = 1. if S > K else 0.
        call_price = S - K if S > K else 0.
    return call_price, delta

def simulate_hedging(results_df, upfront_fee, loan_tenor, risk_free_rate, spread=.0, trading_fee=.0):
    # Pre-allocate memory for new columns
    hedge_results = pd.DataFrame(index=results_df.index)
    columns = ['unclaimed_collateral', 'delta_hedge_inventory',
               'final_collateral_position', 'hedge_cumulative_pnl', 'hedge_unwind_pnl',
               'final_hedged_pnl', 'final_hedged_pnl_with_repayment_and_upfront_fee',
               'final_unhedged_pnl_with_repayment_and_upfront_fee', 'delta_min', 'delta_max', 'delta_median', 'delta_25_percentile', 'delta_75_percentile']
    for col in columns:
        hedge_results[col] = np.nan

    # Assuming results_df is already populated with the necessary columns
    # including 'price', 'volatility', and 'repayment_amounts'
    for i, from_row in results_df.iterrows():
        K = from_row['repayment_amount'] / (1 - upfront_fee)  # Strike price
        
        # Initialize variables for tracking the hedge
        delta_hedge_inventory = 0  # How much of the underlying we currently hold
        cumulative_hedge_pnl = 0  # Total cost spent on hedging
        
        delta_values = []

        # Iterate over the subsequent rows starting from the current index `i`
        # Only consider rows up to the loan tenure
        for j in range(i, i + loan_tenor):
            if j >= len(results_df):
                break  # If j exceeds the number of rows, break the loop
            
            to_row = results_df.iloc[j]
            S = to_row['price']  # Starting price of the underlying
            t = loan_tenor - (j - i)
            
            # Calculate call price and delta for the day
            _, delta = black_scholes_call(S, K, t, risk_free_rate, to_row['volatility'])
            
            covered_call_delta = 1 - delta
            
            # Calculate the hedge adjustment needed (delta_hedge_inventory is negative for short positions)
            delta_exposure = covered_call_delta + delta_hedge_inventory
            delta_values.append(covered_call_delta)
            if abs(delta_exposure) > 0:
                # Update inventory and calculate the cost of the hedge adjustment
                if delta_exposure > 0:
                    # sell underlying
                    amount_to_short = delta_exposure
                    hedge_pnl = amount_to_short * S * (1 - spread - trading_fee) # selling is profit (record as negative value)
                    delta_hedge_inventory -= amount_to_short
                else:
                    # buy underlying
                    amount_to_buy = abs(delta_exposure)
                    hedge_pnl = -amount_to_buy * S * (1 + spread + trading_fee) # buying is cost (record as positive value)
                    delta_hedge_inventory += amount_to_buy
                cumulative_hedge_pnl += hedge_pnl
            # Update the underlying price for the next simulation step if needed
            # This would involve moving S to the next price in the price path

        borrower_repays = from_row['price_at_expiry'] * (1 - upfront_fee) > from_row['repayment_amount']
        # Final amounts
        unclaimed_collateral = 0. if borrower_repays else 1.
        final_collateral_position = unclaimed_collateral + delta_hedge_inventory
        # Finally unwind hedge
        hedge_unwind_pnl = final_collateral_position * from_row['price_at_expiry'] * (1 - spread - trading_fee) if final_collateral_position > 0 else final_collateral_position * from_row['price_at_expiry'] * (1 + spread + trading_fee)

        # Record the results in the DataFrame
        hedge_results.loc[i, 'delta_min'] = np.min(delta_values)
        hedge_results.loc[i, 'delta_max'] = np.max(delta_values)
        hedge_results.loc[i, 'delta_median'] = np.median(delta_values)
        hedge_results.loc[i, 'delta_25_percentile'] = np.percentile(delta_values, 25)
        hedge_results.loc[i, 'delta_75_percentile'] = np.percentile(delta_values, 75)
        hedge_results.loc[i, 'unclaimed_collateral'] = unclaimed_collateral
        hedge_results.loc[i, 'delta_hedge_inventory'] = delta_hedge_inventory
        hedge_results.loc[i, 'final_collateral_position'] = final_collateral_position
        hedge_results.loc[i, 'hedge_cumulative_pnl'] = cumulative_hedge_pnl
        hedge_results.loc[i, 'hedge_unwind_pnl'] = hedge_unwind_pnl
        hedge_results.loc[i, 'final_hedged_pnl'] = hedge_unwind_pnl + cumulative_hedge_pnl 
        hedge_results.loc[i, 'final_hedged_pnl_with_repayment_and_upfront_fee'] = hedge_unwind_pnl + cumulative_hedge_pnl + (from_row['repayment_amount'] if borrower_repays else 0.) - from_row['loan_per_coll'] + upfront_fee * from_row['price']
        hedge_results.loc[i, 'final_unhedged_pnl_with_repayment_and_upfront_fee'] = (from_row['repayment_amount'] if borrower_repays else unclaimed_collateral * from_row['price_at_expiry']) - from_row['loan_per_coll'] + upfront_fee * from_row['price']

    # Combine the hedge results with the results_df
    final_results = pd.concat([results_df, hedge_results], axis=1)
    return final_results

def simulate_strategy(df, price_column_name, loan_tenor, ltv_ratio, upfront_fee, APR):
    # Create a copy of the DataFrame to ensure the original DataFrame is not modified
    df_copy = df.copy()

    # Reset index
    df_copy = df_copy.reset_index()

    # Pre-compute values
    loan_values = df_copy[price_column_name] * ltv_ratio
    start_prices = df_copy[price_column_name].values
    end_indices = df_copy.index + loan_tenor
    
    # Ensure indices are within bounds
    valid_indices = end_indices < len(df_copy)
    end_indices = end_indices[valid_indices]
    loan_values = loan_values[valid_indices]
    start_prices = start_prices[valid_indices]
    
    # Fetch end prices using vectorized operations
    end_prices = df_copy.loc[end_indices, price_column_name].values
    
    # Calculate relevant quantities
    repayment_amounts = loan_values * (1 + APR * loan_tenor / 365)
    upfront_amounts = upfront_fee * start_prices
    roi_from_upfront_fee = upfront_amounts / loan_values

    # Initialize results with default values
    defaulted = end_prices * (1 - upfront_fee) < repayment_amounts

    roi_from_interest = repayment_amounts / loan_values - 1
    roi_from_interest[defaulted] = 0

    # Initialize the loss array with zeros
    loss_given_default = np.zeros_like(end_prices)

    # Calculate loss only for cases where the loan defaulted
    loss_given_default[defaulted] = np.minimum(end_prices[defaulted] / loan_values[defaulted] - 1, 0)

    # Construct results DataFrame
    results_df = pd.DataFrame({
        'price': start_prices,
        'price_at_expiry': end_prices,
        'loan_per_coll': loan_values,
        'repayment_amount': repayment_amounts,
        'volatility': df_copy.loc[valid_indices, 'volatility'],
        'loan_inception_time': df_copy.loc[valid_indices, 'snapped_at_datetime'],
        'loan_expiry_time': df_copy.loc[valid_indices, 'snapped_at_datetime'] + pd.Timedelta(days=loan_tenor),
        'defaulted': defaulted,
        'loss_given_default': loss_given_default,
        'roi_from_interest': roi_from_interest,
        'roi_from_upfront_fee': roi_from_upfront_fee,
        'net_roi': loss_given_default + roi_from_interest + roi_from_upfront_fee
    })
    
    return results_df

def get_available_currencies():
    csv_files = [file for file in os.listdir() if file.endswith('.csv')]
    return [file.split('.')[0].upper() for file in csv_files]

def read_and_process_data(collateral_currency, loan_currency, selected_from_date, selected_to_date):
    # Define your date range here
    date_from = pd.to_datetime(selected_from_date)  # Start date
    date_to = pd.to_datetime(selected_to_date)      # End date

    df_collateral = get_historical_data_cg(collateral_currency, 'usd', date_from, date_to)
    df_loan = get_historical_data_cg(loan_currency, 'usd', date_from, date_to)

    df = df_collateral.merge(df_loan, on='timestamp', suffixes=('_coll', '_loan'))

    df['price_coll'] = df_collateral['price']
    df['price_loan'] = df_loan['price'] 

    df['price_coll_vs_usd'] = df_collateral['price']

    df_eth = get_historical_data_cg('ethereum', 'usd', date_from, date_to)
    df = df.merge(df_eth, on='timestamp', suffixes=('_eth', '_non_eth'))
    df['price_eth_vs_usd'] = df_eth['price']

    df['price'] = df['price_coll'] / df['price_loan']  # Cross price

    df['snapped_at_datetime'] = pd.to_datetime(df['timestamp'])
    return df

def format_two_significant_digits(num, _=None):
    if num == 0:
        return "0.00"
    magnitude = int(np.floor(np.log10(abs(num))))
    if magnitude >= 1:
        return f"{num:.2f}"
    else:
        format_str = "{:." + str(2 - magnitude) + "f}"
        return format_str.format(num)

def percentage_formatter(x, pos):
    return f"{100 * x:.2f}%"

def count_formatter(x, pos):
    return f"{int(x):,}"

def plot_price_over_time(df, selected_from_date, selected_to_date, collateral_currency, loan_currency):
    df_filtered = df[(df['snapped_at_datetime'] >= pd.to_datetime(selected_from_date)) & (df['snapped_at_datetime'] <= pd.to_datetime(selected_to_date))]
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 20), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    ax1.fill_between(pd.to_datetime(df_filtered['snapped_at_datetime']), df_filtered['price'], color="skyblue", label='Price', alpha=0.5)
    ax1.plot(pd.to_datetime(df_filtered['snapped_at_datetime']), df_filtered['price'], color='blue')
    ax1.set_title('Price Over Time: {}/{}'.format(collateral_currency, loan_currency), fontsize=22, fontweight="bold")
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
    span = 90
    df_filtered['volatility'] = df_filtered['log_returns'].ewm(span=span).std() * (365**0.5)

    ax2.plot(df_filtered['snapped_at_datetime'], df_filtered['volatility']*100, color='red', label='Volatility')
    ax2.fill_between(df_filtered['snapped_at_datetime'], df_filtered['volatility']*100, color="red", alpha=0.3)
    ax2.set_title('Annualized Volatility: {} vs {} \n ({}-Day EWM Rolling Window)'.format(collateral_currency, loan_currency, span), fontsize=20, fontweight="bold")
    ax2.set_ylabel('Volatility (in %)', fontsize=18, fontweight="bold")
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Annotations and lines for high, low, and latest volatility on ax2 (Volatility plot)
    highest_volatility = df_filtered['volatility'].max()*100
    lowest_volatility = df_filtered['volatility'].min()*100
    last_volatility = df_filtered['volatility'].iloc[-1]*100

    # Check proximity for Volatility annotations
    volatility_threshold = 0.1 * (highest_volatility - lowest_volatility)
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

    # Calculate log returns for both the underlying and ETH prices
    df_filtered['underlying_log_returns'] = np.log(df_filtered['price_coll_vs_usd'] / df_filtered['price_coll_vs_usd'].shift(1))
    df_filtered['eth_log_returns'] = np.log(df_filtered['price_eth_vs_usd'] / df_filtered['price_eth_vs_usd'].shift(1))
    # Drop any NaN values that may have been introduced by our log return calculation
    df_filtered = df_filtered.dropna()
    # Calculate a 30-day rolling correlation
    window_size = 60  # The window size can be adjusted depending on the data frequency and needs
    df_filtered['rolling_correlation'] = df_filtered['underlying_log_returns'].rolling(window=window_size).corr(df_filtered['eth_log_returns'])
    # Calculate rolling covariance between the underlying asset and ETH
    rolling_covariance = df_filtered['underlying_log_returns'].rolling(window=window_size).cov(df_filtered['eth_log_returns'])

    # Calculate rolling variance for ETH
    rolling_variance = df_filtered['eth_log_returns'].rolling(window=window_size).var()

    # Calculate rolling beta for the underlying asset (beta = covariance / variance)
    df_filtered['rolling_beta'] = (rolling_covariance / rolling_variance).round(2)

    # Add rolling correlation to the third axis (ax3)
    ax3.plot(df_filtered['snapped_at_datetime'], df_filtered['rolling_correlation'], label='30-day Rolling Correlation', color='purple')
    ax3.set_title(f'Correlation: {collateral_currency} vs ETH\n(30-Day Rolling Window)', fontsize=20, fontweight='bold')
    ax3.set_ylabel('Rolling Correlation', fontsize=18, fontweight='bold')
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Add a horizontal line at correlation 0 for reference
    ax3.fill_between(df_filtered['snapped_at_datetime'], df_filtered['rolling_correlation'], 0, 
                    where=df_filtered['rolling_correlation'] > 0, color='green', alpha=0.3,
                    interpolate=True)
    ax3.fill_between(df_filtered['snapped_at_datetime'], df_filtered['rolling_correlation'], 0, 
                    where=df_filtered['rolling_correlation'] < 0, color='red', alpha=0.3,
                    interpolate=True)
    
    ax4.plot(df_filtered['snapped_at_datetime'], df_filtered['rolling_beta'], label='30-day Rolling Beta', color='purple')
    ax4.set_title(f'Beta: {collateral_currency} vs ETH\n(30-Day Rolling Window)', fontsize=20, fontweight='bold')
    ax4.set_ylabel('Rolling Beta', fontsize=18, fontweight='bold')
    ax4.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax4.axhline(1, color='black', linewidth=0.5, linestyle='--')  # Beta of 1 line for reference
    ax4.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Beta of 0 line for reference
    margin = 0.1 * (df_filtered['rolling_beta'].max() - df_filtered['rolling_beta'].min())
    y_min = df_filtered['rolling_beta'].min() - margin
    y_max = df_filtered['rolling_beta'].max() + margin
    ax4.set_ylim(y_min, y_max)
    
    ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax4.tick_params(axis='both', which='major', labelsize=14)


    fig.tight_layout()
    fig.autofmt_xdate()
    
    return fig, df_filtered

def main():    
    # Sidebar for input selection
    token_dropdown_names, token_dropdown_names_to_id = get_available_tokens()
    
    with st.sidebar:
        with st.expander("**Asset Pair**", expanded=True):
            # Check if 'RPL' and 'USD' are in the list
            default_collateral_index = token_dropdown_names.index("Ethereum (ETH)")
            default_loan_index = token_dropdown_names.index("USDC (USDC)")

            collateral_currency_dropdown = st.selectbox("Select Collateral Token", token_dropdown_names, index=default_collateral_index)
            loan_currency_dropdown = st.selectbox("Select Loan Token", token_dropdown_names, index=default_loan_index)

            collateral_currency = token_dropdown_names_to_id[collateral_currency_dropdown]['symbol']
            loan_currency = token_dropdown_names_to_id[loan_currency_dropdown]['symbol']

            collateral_currency_id = token_dropdown_names_to_id[collateral_currency_dropdown]['id']
            loan_currency_id = token_dropdown_names_to_id[loan_currency_dropdown]['id']

            today = datetime.today()
            from_date = today - timedelta(days=500)
            
            selected_from_date, selected_to_date = st.date_input("Select Date Range", [from_date, today])
            
            df = read_and_process_data(collateral_currency_id, loan_currency_id, selected_from_date, selected_to_date)

            # Set the min and max date from the dataframe df
            min_date = df['snapped_at_datetime'].min()
            max_date = df['snapped_at_datetime'].max()

        with st.expander("**Your Loan Terms**", expanded=True):
            ltv_detailed = st.number_input("LTV:", min_value=0.01, max_value=1.0, value=.3, format="%.4f")
            max_tenor = (selected_to_date - selected_from_date).days
            tenor_detailed = st.number_input("Loan Tenor (days):", min_value=1, max_value=min([max_tenor, 365]), value=90)
            apr_detailed = st.number_input("APR:", min_value=0.0, max_value=1.0, value=0.02, format="%.4f")
            upfront_fee_detailed = st.number_input("Upfront Fee:", min_value=0.0, max_value=100.0, value=0.005, format="%.4f")

    # Plot the time series at the top
    time_series_fig, df_filtered = plot_price_over_time(df, selected_from_date, selected_to_date, collateral_currency, loan_currency)

    st.write(f"## Lending Backtest for {loan_currency}/{collateral_currency}")
    st.write(f"You can backtest a lending strategy where you would've continuously loaned out {loan_currency} against {collateral_currency} collateral at given loan terms.")
    st.pyplot(time_series_fig)
    st.write(f"The currently selected backtesting period spans from {min_date.strftime('%B %d, %Y')} to {max_date.strftime('%B %d, %Y')} and includes {len(df_filtered)} price observations. During this period, the highest {loan_currency}/{collateral_currency} price was {df_filtered['price'].max():.4f} {loan_currency}, the lowest was {df_filtered['price'].min():.4f} {loan_currency}, and the most recent was {df_filtered.tail(1).price.values[0]:.4f} {loan_currency}.\n\n*In the backtest, it is assumed that you would've loaned out {loan_currency} every day at an LTV of {ltv_detailed*100:.4f}% for {tenor_detailed} days, charging an APR of {apr_detailed*100:.4f}% (to be paid in {loan_currency}) and an upfront fee of {upfront_fee_detailed*100:.4f}% (to be paid in {collateral_currency} on the pledged collateral amount), which equates to {upfront_fee_detailed/ltv_detailed*365/tenor_detailed*100:.4f}% per annum. Note that the backtest assumes that there always would've been a borrower willing to borrow from you at the given terms. Hence, it is crucial to specify loan terms that are realistic and in line with the market.*")

    results_df = simulate_strategy(df_filtered, 'price', tenor_detailed, ltv_detailed, upfront_fee_detailed, apr_detailed)
    results_df['cumulative_roi'] = results_df['net_roi'].cumsum()

    st.markdown(generate_csv_download_link(results_df), unsafe_allow_html=True)

    # Define axes positions
    left = 0.12
    bottom = 0.1
    width = 0.75
    height = 0.8

    # --- Plot 1: Price at loan origination and expiry ---
    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    plt.plot(results_df['loan_inception_time'], results_df['price'], label=f'{loan_currency}/{collateral_currency} Price at Loan Origination', color='black', linewidth=1.5)
    plt.plot(results_df['loan_inception_time'], results_df['price_at_expiry'], label=f'{loan_currency}/{collateral_currency} Price at Loan Expiry', color='blue', alpha=0.7, linewidth=.5, linestyle=':')
    plt.plot(results_df['loan_inception_time'], results_df['loan_per_coll'], label=f'Amount of {loan_currency} loaned per pledged {collateral_currency} Collateral Unit', color='green', alpha=0.7, linewidth=.5)
    plt.vlines(results_df[results_df['defaulted']]['loan_inception_time'], results_df[results_df['defaulted']]['price_at_expiry'], results_df[results_df['defaulted']]['loan_per_coll'], color='red', alpha=0.2, label=f'Defaults and {collateral_currency} Shortfalls', zorder=5)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('{} Price (in {})'.format(collateral_currency, loan_currency), fontsize=14)
    plt.title(f'{loan_currency}/{collateral_currency} Prices', fontsize=18, fontweight='bold')
    plt.legend(fontsize=12, loc='lower center', bbox_to_anchor=(0.5, -.22), ncol=2)
    plt.xlim(min_date, max_date)
    plt.grid(True)
    ax1.set_position([left, bottom, width, height])
    st.write(f"## {loan_currency}/{collateral_currency} Prices at Loan Start and Expiry")
    st.write(f"Below you can see the price evolution of {loan_currency}/{collateral_currency} for the backtesting period, showing both the price at origination of each given loan as well as the price at expiry of the loan. Moreover, the chart also shows the amount of {loan_currency} loaned per pledged {collateral_currency} collateral unit over time, assuming you would've continuously loaned out every day at the given target LTV of {ltv_detailed*100:.4f}%.")
    st.pyplot(plt)
    st.write(f"In the given scenario, you would've underwritten {len(results_df)} loans, of which {len(results_df) - results_df['defaulted'].sum()} ({(len(results_df) - results_df['defaulted'].sum())/len(results_df)*100:.2f}%) would've been repaid because the price at loan expiry would've been higher than the owed repayment amount and {results_df['defaulted'].sum()} ({results_df['defaulted'].sum()/len(results_df)*100:.2f}%) would've defaulted.")
             
    st.write("## Lending RoI Over Time")
    st.write(f"Below, you can see how the loan underwriting would've performed over time. Your overall RoI is comprised of the following:")
    st.write(f"- **Earnings from Repayments:** You would've earned {apr_detailed*100:.4f}% on every successfully repaid loan.")
    st.write(f"- **Earnings from Upfront Fees:** You would've received {upfront_fee_detailed*100:.4f}% upfront on the pledged collateral, regardless of whether the borrower would've later repaid or not.")
    st.write(f"- **Potential Shortfalls:** In case the borrower doesn't repay, you would've received the corresponding {collateral_currency} collateral amount, which will have depreciated in value. The exact loss would depend on how significantly the collateral price would've fallen at the expiry of the loan.")

    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    plt.plot(results_df['loan_inception_time'], np.zeros_like(results_df['loan_inception_time']), color='black', linewidth=0.5)
    plt.plot(results_df['loan_inception_time'], results_df['net_roi'], color='navy', label='Net RoI', linewidth=0.5)
    plt.fill_between(results_df['loan_inception_time'], 0, results_df['net_roi'], where=results_df['net_roi'] >= 0, facecolor='lightgreen', interpolate=True)
    plt.fill_between(results_df['loan_inception_time'], 0, results_df['net_roi'], where=results_df['net_roi'] < 0, facecolor='lightcoral', interpolate=True)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Net RoI (%)', fontsize=14)
    plt.title('Net RoI', fontsize=18, fontweight='bold')
    plt.legend(fontsize=12)
    plt.xlim(min_date, max_date)
    plt.grid(True)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(percentage_formatter))
    ax1.set_position([left, bottom, width, height])
    st.pyplot(plt)
    st.write(f" In the given scenario, your average RoI would've been {results_df['net_roi'].mean()*100:.4f}% ({results_df['net_roi'].mean()*100*365/tenor_detailed:.4f}% p.a.), your best RoI would've been {results_df['net_roi'].max()*100:.4f}% ({results_df['net_roi'].max()*100*365/tenor_detailed:.4f}% p.a.), and your worst {results_df['net_roi'].min()*100:.4f}% ({results_df['net_roi'].min()*100*365/tenor_detailed:.4f}% p.a.).")
    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    plt.plot(results_df['loan_inception_time'], np.zeros_like(results_df['loan_inception_time']), color='black', linewidth=0.5)
    plt.fill_between(results_df['loan_inception_time'], results_df['roi_from_upfront_fee']+results_df['roi_from_interest'], color='lightgreen', label='From Interest', where=(results_df['roi_from_interest'] > 0))
    plt.fill_between(results_df['loan_inception_time'], results_df['roi_from_upfront_fee'], color='forestgreen', label='From Upfront Fee')
    plt.ylabel('Earnings RoI (%)', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.title('Earnings Split', fontsize=18, fontweight='bold')
    plt.legend(fontsize=12)
    plt.xlim(min_date, max_date)
    plt.grid(True)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(percentage_formatter))
    ax1.set_position([left, bottom, width, height])
    st.write("## Earnings: Detailed View")
    tmp_total_earnings = results_df['roi_from_upfront_fee'].sum() + results_df['roi_from_interest'].sum()
    st.write(f"Below, you can see how your earnings are split between {loan_currency} interest payments (based on your provided APR) on successfully repaid loans and upfront fees in {collateral_currency} (based on your provided upfront fee).")
    if tmp_total_earnings > 0:
        interest_percentage = results_df['roi_from_interest'].sum() / tmp_total_earnings * 100
        fee_percentage = results_df['roi_from_upfront_fee'].sum() / tmp_total_earnings * 100
        
        if interest_percentage == 100:
            st.write(f"In your case, all earnings would've come from interest payments.")
        elif fee_percentage == 100:
            st.write(f"In your case, all earnings would've come from upfront fees, which you would've earned regardless of whether a borrower repaid or not.")
        else:
            st.write(f"In your case, interest payments would've constituted {interest_percentage:.2f}% of overall earnings, and upfront fees would've represented {fee_percentage:.2f}% of overall earnings.")
    else:
        st.write(f"In your case, you set both the APR and upfront fee to zero; hence, there's no additional breakdown.")
    st.pyplot(plt)


    # --- Combined Plot: Losses Over Time and Risk Metrics ---
    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()

    # Plot losses over time
    plt.plot(results_df['loan_inception_time'], np.zeros_like(results_df['loan_inception_time']), color='black', linewidth=0.5)
    tmp_losses = results_df['loss_given_default'] + results_df['roi_from_upfront_fee']
    tmp_losses[tmp_losses > 0] = 0
    plt.fill_between(results_df['loan_inception_time'], tmp_losses, color='lightcoral', where=(tmp_losses < 0), label='Losses Over Time')

    # Calculate and plot risk metrics
    percentile_99 = tmp_losses[tmp_losses < 0].quantile(.01)
    percentile_95 = tmp_losses[tmp_losses < 0].quantile(.05)
    percentile_90 = tmp_losses[tmp_losses < 0].quantile(.1)
    worst_case_loss = tmp_losses.min()
    exp_loss_given_shortfall = tmp_losses[tmp_losses < 0].mean()

    # Extract corresponding underwriting dates
    worst_case_date = results_df.loc[tmp_losses == worst_case_loss, 'loan_inception_time'].iloc[0].strftime('%Y-%m-%d')

    # Labels, values, and colors
    labels = ["VaR (99%)", "VaR (95%)", "VaR (90%)", "Worst Case Loss", "Expected Loss Given Shortall"]
    values = [percentile_99, percentile_95, percentile_90, worst_case_loss, exp_loss_given_shortfall]
    colors = ['green', 'orange', 'purple', 'red', 'violet']
    # Zip labels, values, and colors together
    zipped = list(zip(labels, values, colors))

    # Sort by values in descending order
    sorted_zip = sorted(zipped, key=lambda x: x[1], reverse=True)

    # Plot risk metrics
    for label, val, color in sorted_zip:
        plt.axhline(y=val, color=color, linestyle='--', linewidth=1.5, label=f'{label}: {val*100:.2f}%')

    # Labels and title
    plt.ylabel('Loss (%)', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.title('Losses and Risk Metrics Over Time', fontsize=18, fontweight='bold')
    plt.legend(fontsize=12, loc='lower center', bbox_to_anchor=(0.5, -.22), ncol=3)
    plt.xlim(min_date, max_date)
    plt.grid(True)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(percentage_formatter))
    ax1.set_position([left, bottom, width, height])

    # Descriptive texts
    st.write("## Losses: Detailed View")
    st.write(f"Below you can see the losses you would've incurred from lending continuously at the given loan terms. Your expected worst-case loss given a shortfall scenario would've been {exp_loss_given_shortfall*100:.2f}% (i.e., this is the average loss you would've incurred in cases where the combined final recoverable {collateral_currency} plus any upfront fee you earned would've been worth less than the {loan_currency} amount you loaned out). Your 90% VaR would've been {percentile_90*100:.2f}%, your 95% VaR {percentile_95*100:.2f}%, your 99% VaR {percentile_99*100:.2f}%, and your worst case loss would've been {worst_case_loss*100:.2f}% (from a loan underwritten on {worst_case_date}).")
    st.pyplot(plt)

    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    plt.plot(results_df['loan_inception_time'], np.zeros_like(results_df['loan_inception_time']), color='black', linewidth=0.5)
    plt.plot(results_df['loan_inception_time'], results_df['defaulted'].cumsum(), color='red', label='Cumulative Defaults', linewidth=1.5)
    plt.fill_between(results_df['loan_inception_time'], results_df['defaulted'].cumsum(), color='lightcoral')
    plt.ylabel('Cumulative Defaults', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.title('Cumulative Defaults Over Time', fontsize=18, fontweight='bold')
    plt.legend(fontsize=12)
    plt.xlim(min_date, max_date)
    plt.grid(True)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(count_formatter))
    ax1.set_position([left, bottom, width, height])

    # Find the dates of the first and last defaults
    first_default_date = results_df.loc[results_df['defaulted'], 'loan_inception_time'].min()
    last_default_date = results_df.loc[results_df['defaulted'], 'loan_inception_time'].max()

    # Write the descriptive texts
    st.write("## Default Count")
    st.write(f"Below you can see the cumulative defaults over time. Overall, {results_df['defaulted'].sum()} loans would've defaulted.")
    if results_df['defaulted'].sum() > 0:
        st.write(f"The first default would've occurred on {first_default_date.strftime('%Y-%m-%d')} and the last default would've occurred on {last_default_date.strftime('%Y-%m-%d')}")
    st.pyplot(plt)


    # --- Plot 5: Cumulative RoI Over Time ---
    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    plt.plot(results_df['loan_inception_time'], results_df['cumulative_roi'], color='purple', label='Cumulative RoI', linewidth=0.5)
    plt.plot(results_df['loan_inception_time'], np.zeros_like(results_df['loan_inception_time']), color='black', linewidth=0.5)
    plt.fill_between(results_df['loan_inception_time'], 0, results_df['cumulative_roi'], where=results_df['cumulative_roi'] >= 0, facecolor='lightgreen', interpolate=True)
    plt.fill_between(results_df['loan_inception_time'], 0, results_df['cumulative_roi'], where=results_df['cumulative_roi'] < 0, facecolor='lightcoral', interpolate=True)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative RoI', fontsize=14)
    plt.title('Cumulative Return on Investment Over Time', fontsize=18, fontweight='bold')
    plt.legend(fontsize=12)
    plt.xlim(min_date, max_date)
    plt.grid(True)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(percentage_formatter))
    ax1.set_position([left, bottom, width, height])
    st.write("## Cumulative RoI")
    st.write("Below, you can see the cumulative RoI that would have been realized if you had continuously underwritten loans at the given terms, with a borrower taking out a loan from you every day.")
    st.pyplot(plt)

    # Display risk metrics as a heatmap
    st.write("## Risk Metrics Heatmaps")
    st.write("Below, you can select various LTV and tenor combinations to quickly assess risk across different combinations.")
    
    # User input for LTV and Tenor
    selected_ltv = st.multiselect("Select LTV (%)", [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100])
    selected_tenor = st.multiselect("Select Tenor (days)", [1, 3, 5, 10, 30, 60, 90, 120, 180, 360])

    # Check if the user has made selections
    if selected_ltv and selected_tenor:
        # Sort LTV in descending order and Tenor in ascending order
        selected_ltv = sorted(selected_ltv, reverse=True)
        selected_tenor = sorted(selected_tenor, reverse=False)

        # Create DataFrame to store risk metrics for each LTV and tenor combination
        df_heatmap_exp_loss_given_shortfall_res = pd.DataFrame(index=selected_ltv, columns=selected_tenor)
        df_heatmap_percentile_90_loss_res = pd.DataFrame(index=selected_ltv, columns=selected_tenor)
        df_heatmap_percentile_95_loss_res = pd.DataFrame(index=selected_ltv, columns=selected_tenor)
        df_heatmap_percentile_99_loss_res = pd.DataFrame(index=selected_ltv, columns=selected_tenor)
        df_heatmap_worst_case_loss_res = pd.DataFrame(index=selected_ltv, columns=selected_tenor)

        # Simulate loans and calculate risk metrics for each combination
        for ltv in selected_ltv:
            for tenor in selected_tenor:
                df_heatmap_simulation_data = simulate_strategy(df_filtered, 'price', tenor, ltv/100., upfront_fee_detailed/100., apr_detailed)
                tmp_losses_2 = df_heatmap_simulation_data['loss_given_default'] + df_heatmap_simulation_data['roi_from_upfront_fee']
                tmp_losses_2[tmp_losses_2 > 0] = 0
                percentile_99 = tmp_losses_2[tmp_losses_2 < 0].quantile(.01)
                percentile_95 = tmp_losses_2[tmp_losses_2 < 0].quantile(.05)
                percentile_90 = tmp_losses_2[tmp_losses_2 < 0].quantile(.1)
                worst_case_loss = tmp_losses_2.min()
                exp_loss_given_shortfall = tmp_losses_2[tmp_losses_2 < 0].mean()

                df_heatmap_exp_loss_given_shortfall_res.loc[ltv, tenor] = np.nan_to_num(exp_loss_given_shortfall)
                df_heatmap_percentile_90_loss_res.loc[ltv, tenor] = np.nan_to_num(percentile_90)
                df_heatmap_percentile_95_loss_res.loc[ltv, tenor] = np.nan_to_num(percentile_95)
                df_heatmap_percentile_99_loss_res.loc[ltv, tenor] = np.nan_to_num(percentile_99)
                df_heatmap_worst_case_loss_res.loc[ltv, tenor] = np.nan_to_num(worst_case_loss)

        # Convert values to float for plotting
        df_heatmap_exp_loss_given_shortfall_res = df_heatmap_exp_loss_given_shortfall_res.apply(pd.to_numeric, errors='coerce')
        df_heatmap_percentile_90_loss_res = df_heatmap_percentile_90_loss_res.apply(pd.to_numeric, errors='coerce')
        df_heatmap_percentile_95_loss_res = df_heatmap_percentile_95_loss_res.apply(pd.to_numeric, errors='coerce')
        df_heatmap_percentile_99_loss_res = df_heatmap_percentile_99_loss_res.apply(pd.to_numeric, errors='coerce')
        df_heatmap_worst_case_loss_res = df_heatmap_worst_case_loss_res.apply(pd.to_numeric, errors='coerce')

        # Custom color palette from red to green
        cmap = sns.diverging_palette(10, 130, as_cmap=True)

        st.write(f"**Expected Loss Given Shortfall:**")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_heatmap_exp_loss_given_shortfall_res, annot=True, cmap=cmap, cbar_kws={'format': ticker.PercentFormatter(1)}, ax=ax)
        plt.xlabel("Tenor (days)")
        plt.ylabel("LTV (%)")
        st.pyplot(fig)

        st.write(f"**VaR Loss (90%):**")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_heatmap_percentile_90_loss_res, annot=True, cmap=cmap, cbar_kws={'format': ticker.PercentFormatter(1)}, ax=ax)
        plt.xlabel("Tenor (days)")
        plt.ylabel("LTV (%)")
        st.pyplot(fig)

        st.write(f"**VaR Loss (95%):**")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_heatmap_percentile_95_loss_res, annot=True, cmap=cmap, cbar_kws={'format': ticker.PercentFormatter(1)}, ax=ax)
        plt.xlabel("Tenor (days)")
        plt.ylabel("LTV (%)")
        st.pyplot(fig)

        st.write(f"**VaR Loss (99%):**")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_heatmap_percentile_99_loss_res, annot=True, cmap=cmap, cbar_kws={'format': ticker.PercentFormatter(1)}, ax=ax)
        plt.xlabel("Tenor (days)")
        plt.ylabel("LTV (%)")
        st.pyplot(fig)

        st.write(f"**Worst Case Loss:**")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_heatmap_worst_case_loss_res, annot=True, cmap=cmap, cbar_kws={'format': ticker.PercentFormatter(1)}, ax=ax)
        plt.xlabel("Tenor (days)")
        plt.ylabel("LTV (%)")
        st.pyplot(fig)
    else:
        st.write("Please select values for LTV and Tenor.")


    # Button to generate plot
    st.write(f"## Covered Call Delta Simulation**")
    if st.button('Simulate Delta Distribution'):
        tmp = simulate_hedging(results_df, upfront_fee_detailed, tenor_detailed, risk_free_rate=0.0, spread=0.0, trading_fee=0.0)

        plt.figure(figsize=(14, 7))
        plt.style.use('seaborn-darkgrid')

        # Representing min, max, and median as points
        plt.scatter(tmp['loan_inception_time'], tmp['delta_min'], label='Min. Delta', color='blue', marker='o')
        plt.scatter(tmp['loan_inception_time'], tmp['delta_max'], label='Max. Delta', color='red', marker='^')
        plt.scatter(tmp['loan_inception_time'], tmp['delta_median'], label='Median Delta', color='green', marker='s')

        # Fill between 25th and 75th percentile
        plt.fill_between(tmp['loan_inception_time'], tmp['delta_25_percentile'], tmp['delta_75_percentile'], color='gray', alpha=0.2, label='25th-75th Percentile Range')

        plt.xlabel('Loan Inception Time', fontsize=14)
        plt.ylabel('Delta Value', fontsize=14)
        plt.title('Distribution of Deltas Across Time', fontsize=16)
        plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
        plt.grid(True)
        plt.tight_layout()

        st.pyplot(plt)

if __name__ == "__main__":
    main()
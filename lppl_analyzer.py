import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from lppls import lppls
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from joblib import Parallel, delayed
import multiprocessing

def _compute_confidence(t2_idx, observations, min_window, max_window, window_step, filter_conditions, max_searches):
    """
    Helper function to compute confidence score for a single time point (t2).
    Executed in parallel.
    """
    # Create a local LPPLS instance for this process
    # Note: LPPLS init is lightweight
    model = lppls.LPPLS(observations=observations)
    
    fits_count = 0
    valid_fits_count = 0
    
    for w_size in range(min_window, max_window + 1, window_step):
        t1_idx = t2_idx - w_size
        if t1_idx < 0:
            continue
        
        # Extract window data: (2, w_size)
        obs_window = observations[:, t1_idx:t2_idx]
        
        try:
            # Fit model
            tc, m, w, a, b, c, c1, c2, O, D = model.fit(max_searches=max_searches, obs=obs_window)
            
            fits_count += 1
            
            # Check filters
            if (filter_conditions['m_min'] <= m <= filter_conditions['m_max'] and
                filter_conditions['omega_min'] <= w <= filter_conditions['omega_max'] and
                b < filter_conditions['B_max'] and
                abs(c) < filter_conditions['C_abs_max']):
                valid_fits_count += 1
        except Exception:
            continue
            
    return valid_fits_count / fits_count if fits_count > 0 else 0

class LPPLAnalyzer:
    def __init__(self, ticker_name, ticker_symbol, start_date=None, end_date=None):
        self.ticker_name = ticker_name
        self.ticker_symbol = ticker_symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.lppls_model = None
        
    def fetch_data(self):
        """Fetches historical data from Yahoo Finance."""
        print(f"Fetching data for {self.ticker_name} ({self.ticker_symbol})...")
        try:
            # Download data
            df = yf.download(self.ticker_symbol, start=self.start_date, end=self.end_date, progress=False)
            
            if df.empty:
                print(f"Error: No data found for {self.ticker_symbol}")
                return False
            
            # Check available columns
            if 'Adj Close' in df.columns:
                self.data = df['Adj Close']
            elif 'Close' in df.columns:
                self.data = df['Close']
            else:
                # Handle MultiIndex case (e.g. ('Adj Close', 'Ticker'))
                # Flatten columns if needed or search recursively
                # Recent yfinance might return columns like (Price, Ticker)
                try:
                    # Try to access by cross-section if multi-index
                    self.data = df.xs('Adj Close', axis=1, level=0)
                except:
                    try:
                        self.data = df.xs('Close', axis=1, level=0)
                    except:
                         print(f"Error: Could not find 'Adj Close' or 'Close' in columns: {df.columns}")
                         return False

            # If self.data is a DataFrame (from xs with one column), squeeze it to Series
            if isinstance(self.data, pd.DataFrame):
                self.data = self.data.squeeze()
                
            # Final check
            if self.data is None or self.data.empty:
                 print(f"Error: Data Series is empty after extraction.")
                 return False
            
            self.data = self.data.dropna()
            
            # Ensure index is datetime
            self.data.index = pd.to_datetime(self.data.index)
            
            print(f"Successfully fetched {len(self.data)} data points.")
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def run_analysis(self, window_size=126, step=1, filter_conditions=None, is_antibubble=False):
        """
        Runs the LPPL analysis (Confidence Indicator) manually and in PARALLEL.
        """
        if self.data is None:
            print("Data not loaded. Call fetch_data() first.")
            return None, None

        print(f"Starting {'Anti-Bubble' if is_antibubble else 'Bubble'} Analysis for {self.ticker_name}...")

        # Prepare data
        price_series = self.data.copy()
        
        if is_antibubble:
            price_series = 1.0 / price_series
        
        # Convert to numpy arrays
        # Use integer index (0, 1, 2...) instead of dates/ordinals to avoid gaps (weekends) issues
        time_ord = np.arange(len(price_series))
        
        # CRITICAL FIX: LPPLS model expects LOG-PRICE
        # We take log of the price (or inverted price for anti-bubble)
        price = np.log(price_series.values)
        
        # Stack time and price (2, N)
        observations = np.array([time_ord, price])
        
        # Define parameters for multi-scale analysis
        min_window = int(window_size / 4) # Allow smaller windows relative to the larger max window
        max_window = window_size
        
        # Remove lookback limit to analyze full history
        total_days = len(price_series)
        start_index = max_window 
        
        print(f"Computing confidence indicators for {total_days - start_index} days (Parallel)...")
        
        # Default filter conditions
        if filter_conditions is None:
            filter_conditions = {
                'm_min': 0.1, 'm_max': 0.9,
                'omega_min': 4, 'omega_max': 25, # Relaxed from [6, 13] to [4, 25]
                'B_max': 0,
                'C_abs_max': 1
            }

        # Prepare arguments for parallel execution
        indices_to_compute = range(start_index, total_days, step)
        window_step = 5
        max_searches = 100
        
        num_cores = multiprocessing.cpu_count()
        print(f"Using {num_cores} cores.")

        # Execute in parallel using joblib
        # We use tqdm to show progress
        try:
            from tqdm import tqdm
            # Use a generator to wrap the parallel execution for tqdm
            results = Parallel(n_jobs=num_cores)(
                delayed(_compute_confidence)(
                    t2_idx, observations, min_window, max_window, window_step, filter_conditions, max_searches
                ) for t2_idx in tqdm(indices_to_compute)
            )
        except ImportError:
            results = Parallel(n_jobs=num_cores)(
                delayed(_compute_confidence)(
                    t2_idx, observations, min_window, max_window, window_step, filter_conditions, max_searches
                ) for t2_idx in indices_to_compute
            )
            
        # Map results back to dates
        dates = [price_series.index[i] for i in indices_to_compute]
            
        # Create Series
        confidence_indicator = pd.Series(results, index=dates)
        
        # Return results and the price series used (inverted or not)
        return confidence_indicator, price_series

    # calculate_confidence_score is now integrated into run_analysis
    # We keep the method signature for compatibility if needed, or just remove it.
    # For now, we remove it to avoid confusion.

    def plot_analysis(self, price_series, confidence_indicator, title, filename):
        """Plots the price and confidence indicator."""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot Price
        # If inverted for anti-bubble, we might want to plot the ORIGINAL price for context,
        # but align the indicator.
        # Let's plot the original price if possible.
        
        # Re-align dates
        # confidence_indicator index is already datetime (set in run_analysis)
        dates = confidence_indicator.index
        
        # Plot Original Price (even for anti-bubble, users want to see the actual price)
        ax1.plot(self.data.index, self.data.values, color='black', label='Price', linewidth=1)
        ax1.set_ylabel('Price')
        ax1.set_title(title)
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.legend()
        
        # Plot Confidence Indicator
        ax2.plot(dates, confidence_indicator.values, color='red', label='LPPLS Confidence', linewidth=1)
        ax2.fill_between(dates, confidence_indicator.values, 0, color='red', alpha=0.3)
        ax2.set_ylabel('Confidence')
        ax2.set_ylim(0, 1.05)
        ax2.axhline(y=0.8, color='darkred', linestyle='--', label='Critical Threshold (0.8)')
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        plt.close()


import os
import pandas as pd
from datetime import datetime
from lppl_analyzer import LPPLAnalyzer

import argparse

def main():
    # Define indices to analyze
    # Format: {'Name': 'Ticker'}
    all_indices = {
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC',
        'Nikkei 225': '^N225',
        'TOPIX 100 (Proxy: TOPIX ETF)': '1306.T', # Using 1306.T (Nomura TOPIX ETF) as ^TPX is unreliable
        'KOSPI': '^KS11',
        'KOSDAQ': '^KQ11'
    }

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Global Market LPPL Bubble/Anti-Bubble Diagnosis')
    parser.add_argument('indices', nargs='*', help='Names of indices to analyze (partial match allowed). If empty, runs all.')
    parser.add_argument('--bubble', action='store_true', help='Run only Bubble analysis')
    parser.add_argument('--anti', action='store_true', help='Run only Anti-Bubble analysis')
    
    args = parser.parse_args()

    # Determine modes
    run_bubble = True
    run_anti = True
    
    if args.bubble and not args.anti:
        run_anti = False
    elif args.anti and not args.bubble:
        run_bubble = False
    # If both or neither, run both

    # Filter indices
    if args.indices:
        indices = {}
        for search_term in args.indices:
            found = False
            for name, ticker in all_indices.items():
                if search_term.lower() in name.lower() or search_term.lower() in ticker.lower():
                    indices[name] = ticker
                    found = True
            if not found:
                print(f"Warning: No index found matching '{search_term}'")
    else:
        indices = all_indices

    if not indices:
        print("No indices selected. Exiting.")
        return

    # Configuration
    start_date = '2005-01-01' # Adjust as needed for historical context
    end_date = datetime.now().strftime('%Y-%m-%d')
    output_dir = 'results'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("==================================================")
    print("      Global Market LPPL Bubble/Anti-Bubble Diagnosis      ")
    print("==================================================")
    print(f"Modes: Bubble={run_bubble}, Anti-Bubble={run_anti}")
    print(f"Indices: {list(indices.keys())}")

    for name, ticker in indices.items():
        print(f"\nProcessing {name} ({ticker})...")
        
        analyzer = LPPLAnalyzer(name, ticker, start_date, end_date)
        
        # 1. Fetch Data
        if not analyzer.fetch_data():
            continue
            
        # 2. Bubble Analysis (Positive Bubble)
        if run_bubble:
            print(f"  -> Running Bubble Analysis...")
            conf_bubble, price_bubble = analyzer.run_analysis(
                window_size=500, # Increased to ~2 years as requested
                is_antibubble=False
            )
            
            analyzer.plot_analysis(
                price_bubble, 
                conf_bubble, 
                f'{name} - Bubble Indicator (LPPLS)', 
                os.path.join(output_dir, f'{name.replace(" ", "_")}_Bubble.png')
            )
        
        # 3. Anti-Bubble Analysis (Negative Bubble)
        if run_anti:
            print(f"  -> Running Anti-Bubble Analysis...")
            conf_anti, price_anti = analyzer.run_analysis(
                window_size=500, # Increased to ~2 years
                is_antibubble=True
            )
            
            analyzer.plot_analysis(
                price_anti, # This is the inverted price used for fit
                conf_anti, 
                f'{name} - Anti-Bubble Indicator (LPPLS)', 
                os.path.join(output_dir, f'{name.replace(" ", "_")}_AntiBubble.png')
            )
        
        print(f"Completed {name}.")

    print("\nAll analyses completed. Check the 'results' directory.")

if __name__ == "__main__":
    main()

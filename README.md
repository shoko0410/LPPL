# Global Market LPPL Analysis

This tool analyzes 6 major global stock indices for Bubbles and Anti-Bubbles using the Log-Periodic Power Law (LPPL) model.

## Indices Analyzed
1. **US**: S&P 500 (`^GSPC`), NASDAQ (`^IXIC`)
2. **Japan**: Nikkei 225 (`^N225`), TOPIX (`^TPX`)
3. **Korea**: KOSPI (`^KS11`), KOSDAQ (`^KQ11`)

## Methodology
- **Bubble Detection**: Fits the LPPL model to log-prices to detect super-exponential growth.
- **Anti-Bubble Detection**: Inverts the price data ($1/p(t)$) and fits the LPPL model to detect super-exponential decline (which looks like growth in inverted data).
- **Confidence Indicator**: Calculates the fraction of valid model fits across multiple time windows (Multi-scale analysis). A score > 0.8 indicates a high probability of a regime change (Crash or Rebound).

## Requirements
- Python 3.8+
- Internet connection (to fetch data from Yahoo Finance)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python main.py
```

### Advanced Usage (CLI Arguments)

You can filter by index name and analysis type.

**1. Run specific indices (partial match):**
```bash
python main.py "S&P" "NASDAQ"
python main.py KOSPI
```

**2. Run only Anti-Bubble Analysis:**
```bash
python main.py --anti
```

**3. Combine filters:**
```bash
# Run Anti-Bubble analysis for S&P 500 only
python main.py "S&P" --anti

# Run Bubble analysis for KOSDAQ
python main.py KOSDAQ --bubble
```

Results (charts) will be saved in the `results/` directory.

## Interpretation
- **Red Line (Confidence)**: The probability of a bubble/anti-bubble signal.
- **Threshold (0.8)**: Signals above this line indicate a critical state.
    - **Bubble Chart**: High confidence -> Risk of Crash.
    - **Anti-Bubble Chart**: High confidence -> Potential for Rebound (Bottom).

## Disclaimer
This tool is for educational and research purposes only. It is not financial advice.

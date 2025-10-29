# Task 1: Natural Gas Price Data Analysis and Extrapolation

## Objective
Analyze monthly natural gas price data to estimate prices at any given date and extrapolate future prices for an additional year, accounting for seasonal trends.

## Business Problem
The commodity trading desk needs to price natural gas storage contracts but lacks sufficient data granularity. Current data consists of monthly snapshots covering approximately 18 months into the future, combined with historical prices. This is insufficient for accurate contract pricing.

## Data Specification

### Input Data
- **File**: `Nat_Gas.xlsx`
- **Format**: CSV/Excel format
- **Date Range**: October 31, 2020 to September 30, 2024
- **Frequency**: Monthly (end of month)
- **Data Points**: Natural gas purchase prices

### Data Characteristics
Each data point represents the market price of natural gas delivered at the end of a calendar month, sourced from a market data provider.

## Requirements

### Functional Requirements
1. Load and process monthly natural gas price data
2. Analyze historical price patterns and seasonal trends
3. Interpolate prices for any date within the historical range
4. Extrapolate prices for one year beyond available data (through September 2025)
5. Return price estimates for any input date

### Technical Requirements
- Handle date inputs in standard formats
- Account for seasonal variations in pricing
- Provide smooth price transitions between data points
- Validate input dates
- Handle edge cases (dates before/after data range)

## Methodology Considerations

### Analysis Phase
- Identify seasonal patterns (winter peaks, summer lows)
- Detect long-term trends
- Analyze price volatility
- Identify any anomalies or outliers

### Interpolation Strategy
For dates between available monthly data points, consider:
- Linear interpolation for short gaps
- Seasonal adjustment factors
- Trend preservation

### Extrapolation Strategy
For future dates beyond available data, account for:
- Historical seasonal patterns
- Long-term price trends
- Cyclic behavior in natural gas markets
- Uncertainty increases with projection distance

## Expected Deliverables

### Core Functionality
A price estimation system that accepts a date and returns an estimated natural gas price.

### Supporting Analysis
- Data exploration and visualization
- Trend and seasonality analysis
- Model validation metrics
- Price forecast visualization

## Assumptions
- No need to account for market holidays or weekends
- Prices change smoothly over time
- Historical seasonal patterns are indicative of future behavior
- No consideration of external factors (weather, geopolitics, supply disruptions)

## Success Metrics
- Accurate interpolation within historical data range
- Reasonable extrapolation reflecting seasonal patterns
- Smooth price curves without artificial discontinuities
- Documented approach and methodology

## Tools and Environment
- **IDE**: VSCode
- **Language**: Python
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Date Handling**: datetime module

## Constraints
- Data limited to monthly snapshots
- No access to intraday or weekly price data
- Must work with publicly available historical data only
- Extrapolation limited to one year beyond available data

## Integration Notes
The output from this task will be used in Task 2 to price commodity storage contracts, where accurate price estimation at injection and withdrawal dates is critical for valuation.
# JP Morgan Chase & Co. - Quantitative Research Virtual Experience

## Overview
This repository contains solutions for the JP Morgan Chase & Co. Quantitative Research virtual work experience program on Forage. The program consists of four tasks covering commodity trading, credit risk analysis, and predictive modeling for retail banking.

## Repository Structure
```
├── Task_1_Natural_Gas_Analysis/
│   ├── task_1_solution.py
│   └── Nat_Gas.xlsx
├── Task_2_Commodity_Storage_Pricing/
│   ├── task_2_solution.py
│   └── Nat_Gas.xlsx
├── Task_3_Credit_Risk_Analysis/
│   ├── task_3_solution.ipynb
│   └── Loan_Data.csv
├── Task_4_FICO_Score_Bucketing/
│   ├── task_4_solution.ipynb
│   └── Loan_Data.csv
└── README.md
```

## Technologies Used
- **Python 3.x**
- **VSCode** - Tasks 1 & 2
- **Jupyter Notebook** - Tasks 3 & 4
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

---

## Task 1: Natural Gas Price Data Analysis and Extrapolation

### Objective
Analyze monthly natural gas price data to estimate prices at any given date and extrapolate future prices for an additional year, accounting for seasonal trends.

### Business Context
The commodity trading desk wants to trade natural gas storage contracts but lacks sufficient data granularity. Storage contracts are agreements between warehouse owners and supply chain participants to store physical commodities for specified periods, allowing traders to capitalize on seasonal price differentials.

### Problem Statement
Current market data consists of monthly snapshots representing natural gas prices delivered at the end of each calendar month. This data covers approximately 18 months into the future but lacks the granularity needed for accurate contract pricing.

### Data Specification
- **File**: `Nat_Gas.xlsx`
- **Date Range**: October 31, 2020 to September 30, 2024
- **Frequency**: Monthly (end of month)
- **Data Points**: Natural gas purchase prices

### Key Requirements
- Load and process monthly natural gas price data
- Analyze historical price patterns and seasonal trends
- Interpolate prices for any date within the historical range
- Extrapolate prices for one year beyond available data
- Return price estimates for any input date

### Deliverables
- Price estimation function accepting a date and returning estimated price
- Data visualization showing trends and seasonality
- Model validation and accuracy metrics

---

## Task 2: Commodity Storage Contract Pricing Model

### Objective
Develop a pricing model for natural gas storage contracts that calculates the contract value based on injection/withdrawal dates, storage costs, and price differentials.

### Business Context
Clients want to buy gas now to store and sell in winter to take advantage of seasonal price increases. The contract value is determined by the difference between purchase and sale prices, minus all associated costs.

### Pricing Formula
```
Contract Value = (Sell Price - Buy Price) × Volume - Total Costs

Where Total Costs include:
- Storage costs (monthly fees)
- Injection costs
- Withdrawal costs
- Transportation costs (if applicable)
```

### Input Parameters
- **Injection dates**: When gas is purchased and stored
- **Withdrawal dates**: When gas is withdrawn and sold
- **Prices**: Purchase/sale prices at those dates (from Task 1)
- **Injection/withdrawal rate**: Rate at which gas can be moved
- **Maximum storage volume**: Storage capacity limit
- **Storage costs**: Periodic fees for storage

### Key Requirements
- Handle multiple injection and withdrawal dates
- Calculate all cash flows involved in the contract
- Support various storage configurations
- Provide contract valuation

### Assumptions
- No transport delay
- Zero interest rates
- No need to account for market holidays, weekends, or bank holidays

### Deliverables
- Pricing function that takes contract parameters and returns value
- Test cases with sample inputs
- Documentation of pricing methodology

---

## Task 3: Credit Risk Analysis and Default Prediction

### Objective
Build a predictive model to estimate the probability of default (PD) for personal loan borrowers and calculate expected loss for the retail banking portfolio.

### Business Context
The retail banking arm is experiencing higher-than-expected default rates on personal loans. Better estimates of default probability will allow the bank to set aside sufficient capital to absorb potential losses.

### Problem Statement
Using historical borrower data (income, outstanding loans, and other metrics), develop a model that predicts the probability a borrower will default on their loan.

### Data Specification
- **File**: `task 3 and task 4_Loan_Data.csv`
- **Format**: Tabular data with borrower characteristics
- **Features**: Income, total loans outstanding, and other financial metrics
- **Target**: Binary indicator of previous default

### Key Requirements
- Train a predictive model using provided borrower data
- Estimate probability of default (PD) for new borrowers
- Calculate expected loss assuming 10% recovery rate
- Compare multiple modeling approaches if possible

### Expected Loss Calculation
```
Expected Loss = Loan Amount × PD × (1 - Recovery Rate)
Where Recovery Rate = 10% (0.1)
```

### Modeling Approaches
Consider various techniques:
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- Neural Networks

### Deliverables
- Trained predictive model
- Function that takes loan properties and outputs expected loss
- Model performance metrics (accuracy, precision, recall, AUC-ROC)
- Comparative analysis of different methods (if applicable)

---

## Task 4: FICO Score Bucketing for Mortgage Risk

### Objective
Develop a quantization technique to bucket FICO scores into categorical ranges that optimize predictive power for mortgage default probability.

### Business Context
FICO scores (ranging from 300 to 850) need to be mapped into discrete buckets for use in a machine learning model that requires categorical inputs. The bucketing strategy should maximize the model's ability to predict defaults.

### Problem Statement
Given FICO scores for mortgage borrowers, create an optimal rating map that assigns scores to buckets where lower ratings signify better credit scores. The approach must be generalizable to future datasets.

### Key Concepts
- **FICO Score**: Credit score from 300-850 quantifying borrower creditworthiness
- **Quantization**: Process of mapping continuous values to discrete buckets
- **Dynamic Programming**: Technique to solve optimization problem incrementally

### Optimization Approaches

#### Mean Squared Error
Minimize the squared difference between actual values and bucket representatives.

#### Log-Likelihood
Maximize the likelihood function considering default density in each bucket, which accounts for both discretization roughness and default concentration.

### Key Requirements
- Create a general approach for generating bucket boundaries
- Optimize bucket properties (MSE or log-likelihood)
- Ensure buckets effectively separate default risk levels
- Map FICO scores to ratings (lower rating = better credit)

### Methodology Considerations
- Split problem into subproblems (dynamic programming)
- Balance bucket size and default prediction power
- Ensure monotonic relationship between FICO and default probability
- Validate bucketing strategy on hold-out data

### Deliverables
- Bucketing algorithm with configurable number of buckets
- Optimal bucket boundaries for the dataset
- Rating map from FICO scores to risk categories
- Visualization of bucket distributions and default rates
- Performance comparison of different optimization approaches

---

## Key Learnings

### Quantitative Finance
- Commodity storage contract valuation
- Understanding of natural gas markets and seasonal pricing
- Cash flow analysis for derivative contracts

### Risk Management
- Credit risk modeling and probability of default estimation
- Expected loss calculation and capital allocation
- FICO score analysis and creditworthiness assessment

### Machine Learning
- Predictive modeling for financial applications
- Feature engineering for credit risk
- Model validation and performance metrics
- Quantization and discretization techniques

### Technical Skills
- Time series analysis and forecasting
- Data preprocessing and feature engineering
- Model selection and hyperparameter tuning
- Dynamic programming for optimization problems

---

## Setup and Installation

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter openpyxl
```

### Running the Code

#### Tasks 1 & 2 (VSCode)
```bash
python Task_1_Natural_Gas_Analysis/solution.py
python Task_2_Commodity_Storage_Pricing/solution.py
```

#### Tasks 3 & 4 (Jupyter Notebook)
```bash
jupyter notebook
# Navigate to respective task folders and open .ipynb files
```

---

## Notes and Observations

### Task 1
Successfully implemented natural gas price analysis with seasonal trend consideration. The model accounts for cyclical patterns in commodity pricing.

### Task 2
Developed a comprehensive pricing model for storage contracts. The function handles multiple injection/withdrawal dates and various cost structures.

### Task 3
Built predictive models for loan default probability. Explored multiple approaches and compared their performance for credit risk assessment.

### Task 4
Task 4 presented challenges in implementing optimal FICO score bucketing. The solution attempts to balance bucket optimization with practical default prediction, though further refinement may be beneficial.

---

## Future Improvements
- Enhance extrapolation accuracy with advanced time series models (ARIMA, Prophet)
- Incorporate external factors (weather, supply data) into gas price predictions
- Expand credit risk model with additional features and ensemble methods
- Refine FICO bucketing algorithm with more sophisticated optimization
- Add comprehensive unit tests and validation frameworks
- Implement real-time data pipeline integration

---

## References
- JP Morgan Chase & Co. Forage Virtual Experience Program
- Quantization techniques and dynamic programming
- Credit risk modeling and Basel framework
- FICO scoring methodology

---

## Contact
For questions or discussions about this project, feel free to reach out or open an issue.

---

## Acknowledgments
Special thanks to JP Morgan Chase & Co. and Forage for providing this educational opportunity to explore quantitative research in finance.

---

## License
This project is for educational purposes as part of the JP Morgan virtual experience program.

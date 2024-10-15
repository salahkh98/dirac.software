
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
BUDGET = 50000
DAYS_TO_EXPIRY = 400
DAYS_TO_SHIP = 30
DAYS_IN_YEAR = 365

# Load data
file_path = 'data/cls_mod.xlsx'
allocation_df = pd.read_excel(file_path, sheet_name='Allocation')
scenarios_df = pd.read_excel(file_path, sheet_name='Scenarios')

# Preprocess data
allocation_df = allocation_df[allocation_df['MARKETPLACE ID'].notna()]
scenarios_df = scenarios_df[scenarios_df['Marketplace_ID'] != 'MARKETPLACE ID']

def calculate_cash_flows(quantities, scenario):
    cash_flows = [-BUDGET]  # Initial investment
    daily_inflows = []
    
    for index, row in allocation_df.iterrows():
        cost = row['Cost of Goods'] * quantities[index]
        freight = row['Freight Cost'] * quantities[index]
        initial_outflow = -(cost + freight)
        cash_flows[0] += initial_outflow
        
        win_rate = float(scenarios_df.loc[scenarios_df['Marketplace_ID'] == row['MARKETPLACE ID'], f'Scenario_{scenario}_Win_Rate'].values[0])
        avg_price = float(scenarios_df.loc[scenarios_df['Marketplace_ID'] == row['MARKETPLACE ID'], f'Scenario_{scenario}_Ave_Price'].values[0])
        
        daily_inflow = (avg_price - row['Shipping fees'] - row['Referral USD']) * win_rate * quantities[index] / DAYS_IN_YEAR
        daily_inflows.append(daily_inflow)
    
    total_daily_inflow = sum(daily_inflows)
    cash_flows.extend([total_daily_inflow] * (DAYS_TO_EXPIRY - DAYS_TO_SHIP))
    
    return np.array(cash_flows)

def npv(rate, cash_flows):
    return np.sum(cash_flows / (1 + rate)**np.arange(len(cash_flows)))

def irr(cash_flows, guess=0.1):
    return minimize(lambda r: npv(r, cash_flows)**2, guess).x[0]

def calculate_irr(cash_flows):
    try:
        return irr(cash_flows)
    except:
        return -1  # Return a negative value if IRR calculation fails

def objective_function(quantities, scenario):
    cash_flows = calculate_cash_flows(quantities, scenario)
    return -npv(0.1, cash_flows)  # Using 10% as discount rate, negative because we want to maximize

def constraint_budget(quantities):
    return BUDGET - np.sum(allocation_df['Cost of Goods'] * quantities) - np.sum(allocation_df['Freight Cost'] * quantities)

def optimize_allocation(scenario):
    initial_guess = np.ones(len(allocation_df)) * (BUDGET / len(allocation_df) / allocation_df['Cost of Goods'].mean())
    bounds = [(0, min(row['Available_Stock'], BUDGET / row['Cost of Goods'])) for _, row in allocation_df.iterrows()]
    
    constraint = {'type': 'ineq', 'fun': constraint_budget}
    
    result = minimize(objective_function, initial_guess, args=(scenario,), method='SLSQP', bounds=bounds, constraints=constraint)
    
    return result.x

# Optimize for each scenario
optimized_quantities = {}
for scenario in range(1, 4):
    optimized_quantities[scenario] = optimize_allocation(scenario)
    logging.info(f"Optimized quantities for Scenario {scenario}: {optimized_quantities[scenario]}")

# Calculate IRR, NPV, and other metrics for each scenario
results = {}
for scenario in range(1, 4):
    quantities = optimized_quantities[scenario]
    cash_flows = calculate_cash_flows(quantities, scenario)
    irr_value = calculate_irr(cash_flows)
    npv_value = npv(0.1, cash_flows)  # Using 10% as discount rate
    total_cost = np.sum(allocation_df['Cost of Goods'] * quantities) + np.sum(allocation_df['Freight Cost'] * quantities)
    total_revenue = np.sum(cash_flows[1:])  # Excluding initial investment
    gross_profit = total_revenue - total_cost
    
    results[scenario] = {
        'IRR': irr_value,
        'NPV': npv_value,
        'Gross Profit': gross_profit,
        'Total Cost': total_cost,
        'Total Revenue': total_revenue,
        'Optimized Quantities': quantities
    }
    
    logging.info(f"Scenario {scenario} results: IRR = {irr_value:.2%}, NPV = ${npv_value:.2f}, Gross Profit = ${gross_profit:.2f}")

# Export results to Excel
with pd.ExcelWriter('stock_deal_results.xlsx') as writer:
    for scenario in range(1, 4):
        df = pd.DataFrame({
            'MARKETPLACE ID': allocation_df['MARKETPLACE ID'],
            'CODE': allocation_df['CODE'],
            'Optimized Quantity': results[scenario]['Optimized Quantities'],
            'Cost of Goods': allocation_df['Cost of Goods'] * results[scenario]['Optimized Quantities'],
            'Freight Cost': allocation_df['Freight Cost'] * results[scenario]['Optimized Quantities'],
            'Expected Sale Price': allocation_df['Expected Sale Price'],
            'Shipping fees': allocation_df['Shipping fees'],
            'Referral USD': allocation_df['Referral USD'],
            'Profit': allocation_df['Profit']
        })
        df['Total Cost'] = df['Cost of Goods'] + df['Freight Cost']
        df.to_excel(writer, sheet_name=f'Scenario {scenario}', index=False)
        
    summary_df = pd.DataFrame({
        'Scenario': range(1, 4),
        'IRR': [results[s]['IRR'] for s in range(1, 4)],
        'NPV': [results[s]['NPV'] for s in range(1, 4)],
        'Gross Profit': [results[s]['Gross Profit'] for s in range(1, 4)],
        'Total Cost': [results[s]['Total Cost'] for s in range(1, 4)],
        'Total Revenue': [results[s]['Total Revenue'] for s in range(1, 4)]
    })
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

logging.info("Results exported to stock_deal_results.xlsx")

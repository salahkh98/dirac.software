# # Importing required libraries
# import streamlit as st
# from langchain_together import ChatTogether
# import os

# # Fetching environment variables for tokens
# # PAGE_ACCESS_TOKEN = os.getenv('PAGE_ACCESS_TOKEN')
# # VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')
# # API_KEY = os.getenv('API_KEY')
# # AI_TOKEN = os.getenv("TOGETHER_API_KEY")

# # Setting up the title of the web application using Streamlit
# st.title("My AI")

# # Initializing the Together AI client with API key
# chat = ChatTogether(
#     together_api_key=st.secrets['TOGETHER_API_KEY'],
#     model="meta-llama/Llama-3-70b-chat-hf",  # Example model
# )

# # Ensuring that the Together AI model is set in the session state; defaulting to 'meta-llama/Llama-3-70b-chat-hf'
# if "togather_model" not in st.session_state:
#     st.session_state["togather_model"] = "meta-llama/Llama-3-70b-chat-hf"

# # Ensuring that there is a message list in the session state for storing conversation history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Displaying each message in the session state using Streamlit's chat message display
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Handling user input through Streamlit's chat input box
# if prompt := st.chat_input("What is up?"):
#     # Appending the user's message to the session state
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Displaying the user's message in the chat interface
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Preparing to display the assistant's response
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()  # Placeholder for assistant's response
#         full_response = ""  # Initializing a variable to store the full response

#         # Generating a response from the Together AI model
#         for message_chunk in chat.stream(
#             input=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],  # Passing conversation history
#         ):
#             # Accumulating the streaming response as it's received
#             full_response += message_chunk.content
#             # Updating the placeholder to show the response as it's being 'typed'
#             message_placeholder.markdown(full_response + "▌")

#         # Updating the placeholder with the final response once fully received
#         message_placeholder.markdown(full_response)


#     # Appending the assistant's response to the session's message list
#     st.session_state.messages.append({"role": "assistant", "content": full_response})
# import pandas as pd
# import numpy as np
# import numpy_financial as npf
# import openpyxl

# # Define products and costs
# products = {
#     'S1': {'cost': 2.33, 'marketplace_id': 'A', 'is_multipack': False},
#     'S2': {'cost': 2.33, 'marketplace_id': 'A', 'is_multipack': False},
#     'S3': {'cost': 3.00, 'marketplace_id': 'B', 'is_multipack': True},
#     'S4': {'cost': 4.00, 'marketplace_id': 'C', 'is_multipack': False},
#     'S5': {'cost': 5.00, 'marketplace_id': 'D', 'is_multipack': False},
#     'S6': {'cost': 1.50, 'marketplace_id': 'E', 'is_multipack': False},
#     'S7': {'cost': 2.00, 'marketplace_id': 'F', 'is_multipack': False},
#     'S8': {'cost': 3.50, 'marketplace_id': 'G', 'is_multipack': True},
#     'S9': {'cost': 4.50, 'marketplace_id': 'H', 'is_multipack': False},
#     'S10': {'cost': 6.00, 'marketplace_id': 'I', 'is_multipack': False},
#     'S11': {'cost': 3.25, 'marketplace_id': 'J', 'is_multipack': True},
#     'S12': {'cost': 2.75, 'marketplace_id': 'K', 'is_multipack': False},
# }

# # Define scenarios
# scenarios = {
#     'Scenario 1': {'win_rate': 0.08, 'avg_price': 20.92},
#     'Scenario 2': {'win_rate': 0.19, 'avg_price': 17.38},
#     'Scenario 3': {'win_rate': 0.22, 'avg_price': 21.50}
# }

# # Budget
# budget = 50000
# lead_time_days = 30
# expiry_days = 400

# # Initialize results DataFrame
# results = pd.DataFrame(columns=['Product ID', 'Quantity', 'Total Cost', 'Total Revenue', 'Expected IRR'])

# # Optimize stock purchases
# for scenario_name, scenario in scenarios.items():
#     total_quantity = {}
    
#     # Iterate over products
#     for product_id, details in products.items():
#         if details['is_multipack']:
#             unit_cost = details['cost'] * 2  # Total cost for multipack
#         else:
#             unit_cost = details['cost']
        
#         # Calculate how many units to buy within the budget
#         quantity_to_buy = budget // unit_cost
        
#         total_cost = quantity_to_buy * unit_cost
#         expected_revenue = (quantity_to_buy * scenario['avg_price']) * scenario['win_rate'] * (expiry_days - lead_time_days)
        
#         # Record results
#         results = results._append({
#             'Product ID': product_id,
#             'Quantity': quantity_to_buy,
#             'Total Cost': total_cost,
#             'Total Revenue': expected_revenue,
#         }, ignore_index=True)

# # Calculate Expected IRR for each product
# for index, row in results.iterrows():
#     cash_flows = [-row['Total Cost']] + [row['Total Revenue'] / expiry_days] * expiry_days
#     expected_irr = npf.irr(cash_flows)
#     results.at[index, 'Expected IRR'] = expected_irr

# # Display the results
# print(results)

# # Save results to Excel
# output_file = "data/SVSL_Offer_Analysis.xlsx"
# with pd.ExcelWriter(output_file) as writer:
#     results.to_excel(writer, sheet_name='Results', index=False)

# print(f"Results saved to {output_file}")
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
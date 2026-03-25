# src/services_detect/fraud_api.py

# 1. Change 'def' to 'async def'
async def get_transaction_data(query):
    # If you are using 'requests' here, you should switch to 'httpx'
    # for true async performance, but for now, making it async
    # allows the 'await' in agent.py to work.

    # Simulating an API call
    data = "Transaction flagged: unusual location or high amount"

    return data

# def get_transaction_data(query):
#     return "Transaction flagged: unusual location or high amount"

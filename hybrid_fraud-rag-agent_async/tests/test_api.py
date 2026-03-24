import requests
import json


def test_fraud_query(query_text):
    url = "http://127.0.0.1:8000/query"
    payload = {"query": query_text}
    headers = {"Content-Type": "application/json"}

    print(f"Sending Query: {query_text}")

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            print("Success!")
            print("Response:", json.dumps(response.json(), indent=2))
        else:
            print(f"Failed with status code: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Connection Error: {e}")


if __name__ == "__main__":
    # Test cases for your Fraud RAG
    test_fraud_query("Check for suspicious patterns in transaction TXN_9988")
    test_fraud_query("What is the fraud risk level for user 'Vinod'?")
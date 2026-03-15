def analyze_risk(transaction, cases):

    fraud_count = 0

    for case in cases:
        if case["label"] == "fraud":
            fraud_count += 1

    if fraud_count >= 2:
        risk = "HIGH"
    elif fraud_count == 1:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "transaction": transaction,
        "risk_level": risk,
        "similar_cases": cases
    }
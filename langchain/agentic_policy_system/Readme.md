insurance-agentic-platform/
│
├── README.md
├── docker-compose.yml
├── .env
├── Makefile
│
├── api-gateway/
├── policy-agent/              # LangGraph agent
├── services/
│   ├── policy-service/
│   ├── quote-service/
│   ├── payment-service/
│   ├── notification-service/
│
├── eventing/
│   ├── kafka-topics/
│   ├── producers/
│   ├── consumers/
│
├── shared/
│   ├── schemas/
│   ├── utils/
│   ├── config/
│
├── infra/
│   ├── docker/
│   ├── k8s/
│   ├── helm/
│
├── observability/
│   ├── langsmith/
│   ├── prometheus/
│   ├── grafana/
│
└── tests/
    ├── integration/
    ├── load/



policy-agent/
├── app.py
├── agent/
│   ├── graph.py
│   ├── state.py
│   ├── nodes/
│   │   ├── quote_node.py
│   │   ├── payment_node.py
│   │   ├── activation_node.py
│
├── tools/
│   ├── quote_tool.py
│   ├── payment_tool.py
│   ├── policy_tool.py
│
├── kafka/
│   ├── consumer.py
│   ├── producer.py
│
├── config/
│   ├── settings.py
│
├── requirements.txt
├── Dockerfile
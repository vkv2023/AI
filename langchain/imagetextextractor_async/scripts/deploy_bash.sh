#!/bin/bash

set -e  # stop on error

echo "Deploying Kubernetes resources..."

# Namespace + config
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml

# Weaviate
kubectl apply -f k8s/weaviate/weaviate-deployment.yaml
kubectl apply -f k8s/weaviate/weaviate-service.yaml

# Jaeger
kubectl apply -f k8s/jaeger/jaeger-deployment.yaml
kubectl apply -f k8s/jaeger/jaeger-service.yaml

# API
kubectl apply -f k8s/api/api-deployment.yaml
kubectl apply -f k8s/api/api-service.yaml

# Prometheus
kubectl apply -f k8s/prometheus/prometheus-config.yaml
kubectl apply -f k8s/prometheus/prometheus-deployment.yaml
kubectl apply -f k8s/prometheus/prometheus-service.yaml

echo "Deployment completed successfully!"

#single command to apply all resources
#kubectl apply -f k8s/
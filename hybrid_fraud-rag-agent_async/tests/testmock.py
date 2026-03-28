import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Adjust the path to allow importing modules from src
# Now import the function to be tested
from fraud_rag.weaviate_client import search_docs
    @patch('fraud_rag.weaviate_client.client')
    @patch('fraud_rag.weaviate_client.get_embedding', new_callable=AsyncMock)

from src.fraud_rag.weaviate_client import search_docs

class TestWeaviateClient(unittest.IsolatedAsyncioTestCase):

    @patch('src.fraud_rag.weaviate_client.client')
    @patch('src.fraud_rag.weaviate_client.get_embedding', new_callable=AsyncMock)
    async def should_return_fraud_events_on_successful_hybrid_search(self, mock_get_embedding, mock_weaviate_client):
        query_text = "suspicious transaction"
        mock_embedding_vector = [0.1, 0.2, 0.3]
        mock_get_embedding.return_value = mock_embedding_vector

        mock_fraud_event_object_1 = MagicMock()
        mock_fraud_event_object_1.properties = {"content": "Fraud event details 1"}
        mock_fraud_event_object_2 = MagicMock()
        mock_fraud_event_object_2.properties = {"content": "Fraud event details 2"}

        mock_response = MagicMock()
        mock_response.objects = [mock_fraud_event_object_1, mock_fraud_event_object_2]

        mock_collection = MagicMock()
        mock_collection.query.hybrid = AsyncMock(return_value=mock_response)

        mock_weaviate_client.is_connected.return_value = True
        mock_weaviate_client.collections.get.return_value = mock_collection

        result = await search_docs(query_text)

        mock_weaviate_client.is_connected.assert_called_once()
        mock_get_embedding.assert_called_once_with(query_text)
        mock_weaviate_client.collections.get.assert_called_once_with("FraudEvent")
        mock_collection.query.hybrid.assert_called_once_with(
            query=query_text,
            vector=mock_embedding_vector,
            limit=5,
            alpha=0.3,
            return_properties=["content"]
        )
    @patch('fraud_rag.weaviate_client.client')
    @patch('fraud_rag.weaviate_client.get_embedding', new_callable=AsyncMock)
        self.assertEqual(result[1].properties["content"], "Fraud event details 2")

    @patch('src.fraud_rag.weaviate_client.client')
    @patch('src.fraud_rag.weaviate_client.get_embedding', new_callable=AsyncMock)
    async def should_return_empty_list_when_no_fraud_events_found(self, mock_get_embedding, mock_weaviate_client):
        query_text = "non-existent event"
        mock_embedding_vector = [0.4, 0.5, 0.6]
        mock_get_embedding.return_value = mock_embedding_vector

        mock_response = MagicMock()
        mock_response.objects = []

        mock_collection = MagicMock()
        mock_collection.query.hybrid = AsyncMock(return_value=mock_response)

        mock_weaviate_client.is_connected.return_value = True
        mock_weaviate_client.collections.get.return_value = mock_collection

        result = await search_docs(query_text)

        mock_weaviate_client.is_connected.assert_called_once()
        mock_get_embedding.assert_called_once_with(query_text)
        mock_weaviate_client.collections.get.assert_called_once_with("FraudEvent")
        mock_collection.query.hybrid.assert_called_once_with(
    @patch('fraud_rag.weaviate_client.client')
    @patch('fraud_rag.weaviate_client.get_embedding', new_callable=AsyncMock)
            alpha=0.3,
            return_properties=["content"]
        )
        self.assertEqual(result, [])

    @patch('src.fraud_rag.weaviate_client.client')
    @patch('src.fraud_rag.weaviate_client.get_embedding', new_callable=AsyncMock)
    async def should_connect_to_weaviate_if_not_already_connected(self, mock_get_embedding, mock_weaviate_client):
        query_text = "new connection test"
        mock_embedding_vector = [0.7, 0.8, 0.9]
        mock_get_embedding.return_value = mock_embedding_vector

        mock_fraud_event_object = MagicMock()
        mock_fraud_event_object.properties = {"content": "Connected fraud event"}

        mock_response = MagicMock()
        mock_response.objects = [mock_fraud_event_object]

        mock_collection = MagicMock()
        mock_collection.query.hybrid = AsyncMock(return_value=mock_response)

        mock_weaviate_client.is_connected.return_value = False
        mock_weaviate_client.connect = AsyncMock()
        mock_weaviate_client.collections.get.return_value = mock_collection

        result = await search_docs(query_text)

    @patch('fraud_rag.weaviate_client.client')
    @patch('fraud_rag.weaviate_client.get_embedding', new_callable=AsyncMock)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].properties["content"], "Connected fraud event")

    @patch('src.fraud_rag.weaviate_client.client')
    @patch('src.fraud_rag.weaviate_client.get_embedding', new_callable=AsyncMock)
    async def should_handle_embedding_generation_failure(self, mock_get_embedding, mock_weaviate_client):
        query_text = "query with embedding error"
        mock_get_embedding.side_effect = Exception("Embedding service unavailable")

        mock_weaviate_client.is_connected.return_value = True
        # Mock collections.get even if it won't be called, to avoid AttributeError if the mock is accessed
    @patch('fraud_rag.weaviate_client.client')
    @patch('fraud_rag.weaviate_client.get_embedding', new_callable=AsyncMock)

        result = await search_docs(query_text)

        mock_get_embedding.assert_called_once_with(query_text)
        self.assertEqual(result, [])

    @patch('src.fraud_rag.weaviate_client.client')
    @patch('src.fraud_rag.weaviate_client.get_embedding', new_callable=AsyncMock)
    async def should_handle_weaviate_query_exception(self, mock_get_embedding, mock_weaviate_client):
        query_text = "query with weaviate error"
        mock_embedding_vector = [0.9, 0.8, 0.7]
        mock_get_embedding.return_value = mock_embedding_vector

        mock_collection = MagicMock()
        mock_collection.query.hybrid = AsyncMock(side_effect=Exception("Weaviate query failed"))

        mock_weaviate_client.is_connected.return_value = True
        mock_weaviate_client.collections.get.return_value = mock_collection

        result = await search_docs(query_text)

        mock_get_embedding.assert_called_once_with(query_text)
        mock_weaviate_client.collections.get.assert_called_once_with("FraudEvent")
        mock_collection.query.hybrid.assert_called_once()
        self.assertEqual(result, [])

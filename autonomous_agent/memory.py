from typing import List
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models.models import PointStruct, VectorParams, PointIdsList, Filter, HasIdCondition
from transformers import AutoTokenizer, AutoModel
from autonomous_agent.schemas import Task


class Encoder:
    def __init__(self, encoder_name):
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.model = AutoModel.from_pretrained(encoder_name)

    def get_hidden_size(self):
        return self.model.config.hidden_size

    def encode(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        outputs = self.model(**inputs)

        # Mean pooling
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings

        vector = mean_pooling(outputs[0], inputs['attention_mask']).detach().numpy().tolist()[0]
        return vector


class DenseRetriever:
    def __init__(self, encoder: Encoder, collection_name: str, qdrant_host: str, qdrant_port: int):
        self.encoder = encoder
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        existing_collections = self.qdrant_client.get_collections().collections
        existing_collections = [i.name for i in existing_collections]
        if self.collection_name not in existing_collections:
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=encoder.get_hidden_size(), distance="Cosine")
            )

    def insert(self, task: Task) -> dict:
        vector = self.encoder.encode(task.result)
        task_dict = task.dict()
        result_id = task_dict.pop("result_id")
        response = self.qdrant_client.upsert(
                                             collection_name=self.collection_name,
                                             points=[
                                                 PointStruct(id=result_id,
                                                             payload=task_dict,
                                                             vector=vector
                                                             )
                                             ]
                                            )
        return response

    def search(self,
               query: str,
               top_k: int,
               ) -> List[dict]:
        vector = self.encoder.encode(query)
        hits = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            with_payload=True,
            top=top_k,
        )
        return [{"task": hit.payload["task"], "score": hit.score} for hit in hits]

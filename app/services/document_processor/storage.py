import boto3
from abc import ABC, abstractmethod
from typing import BinaryIO


class StorageBackend(ABC):
    """Abstract base class for document storage"""
    @abstractmethod
    async def store_document(self, file: BinaryIO, filename: str) -> str:
        """Store document and return its identifier"""
        pass
    
    @abstractmethod
    async def get_document(self, identifier: str) -> bytes:
        """Retrieve document content"""
        pass

class S3StorageBackend(StorageBackend):
    def __init__(self, bucket: str = "documents"):
        self.bucket = bucket
        self.s3_client = boto3.client(
            's3',
            endpoint_url='http://localhost:4566',
            aws_access_key_id='test',
            aws_secret_access_key='test',
            region_name='us-east-1'
        )
    
    async def store_document(self, file: BinaryIO, filename: str) -> str:
        s3_key = f"documents/{filename}"
        self.s3_client.upload_fileobj(file, self.bucket, s3_key)
        return s3_key
    
    async def get_document(self, identifier: str) -> bytes:
        response = self.s3_client.get_object(Bucket=self.bucket, Key=identifier)
        return response['Body'].read()

import boto3
from abc import ABC, abstractmethod
from typing import BinaryIO
from config import AWS_ENDPOINT_URL, AWS_ACCESS_KEY_SECRET, AWS_ACCESS_KEY_ID


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
    def __init__(self, bucket: str = "rag"):
        self.bucket = bucket
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=AWS_ENDPOINT_URL,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_ACCESS_KEY_SECRET,
        )

    async def store_document(self, file: BinaryIO, filename: str) -> str:
        s3_key = f"{filename}"
        try:
            self.s3_client.upload_fileobj(file, self.bucket, s3_key)
        except Exception as e:
            print(f"Error uploading file to S3: {e}")
            raise
        return s3_key

    async def get_document(self, identifier: str) -> bytes:
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=identifier)
            return response["Body"].read()
        except Exception as e:
            print(f"Error retrieving file from S3: {e}")
            raise

import os
from dotenv import load_dotenv
from sqlmodel import SQLModel, create_engine, Session

# Load environment variables from .env file
load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "vectordb")



AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "rag")
AWS_ACCESS_KEY_SECRET = os.getenv("AWS_ACCESS_KEY_SECRET", "key")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "http://localhost:9000")

# Read environment variables from .env file


# Construct the database connection string
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Database configuration
engine = create_engine(DATABASE_URL)

# Create tables
SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
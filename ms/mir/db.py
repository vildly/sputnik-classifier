from typing import Any, Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from pymongo.results import InsertOneResult


# Global variable to store the client instance
CLIENT = None


def connect_to_database(connection_string: str | None) -> None:
    global CLIENT
    if connection_string is None:
        raise Exception("Connecting string is empty")
    if CLIENT is not None:
        raise Exception("Client has already been established")

    try:
        # Create the client with a short timeout to detect connectivity issues
        CLIENT = MongoClient(connection_string, serverSelectionTimeoutMS=10_000)
        # Issue a ping command to check if the connection is valid
        CLIENT.admin.command("ping")
    except PyMongoError as exc:
        CLIENT = None
        raise Exception("Failed to connect to MongoDB") from exc


def get_collection(db: str, collection: str) -> Collection:
    if CLIENT is None:
        raise Exception(
            "Database client is not connected. Call connect_to_database first."
        )
    database = CLIENT.get_database(name=db)
    return database.get_collection(name=collection)


def add_one(col: Collection, data: dict) -> InsertOneResult:
    try:
        return col.insert_one(data)
    except Exception as exc:
        raise Exception("Unable to add the data") from exc


def find_one(col: Collection, query: dict) -> Optional[Any]:
    try:
        return col.find_one(query)
    except Exception as exc:
        raise Exception("Unable to find the document") from exc

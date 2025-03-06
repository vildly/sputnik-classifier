from typing import Any, Literal, Optional, Union

from pylo import get_logger

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from pymongo.results import InsertOneResult, UpdateResult


# Global variable to store the client instance
CLIENT = None


logger = get_logger()


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
        res = col.insert_one(data)
        logger.info(f"MONGO: Add: {res.inserted_id}")
        return res
    except Exception as exc:
        raise ValueError("Unable to add the data") from exc


def update_by_id(
    col: Collection,
    doc_id: Union[str, ObjectId],
    update_data: dict,
    upsert: bool = False,
    operator: Literal["$set", "$push", "$inc", "$unset"] = "$set",
) -> UpdateResult:
    try:
        # Convert doc_id to an ObjectId if it is provided as a string
        oid = doc_id if isinstance(doc_id, ObjectId) else ObjectId(doc_id)
        result = col.update_one({"_id": oid}, {operator: update_data}, upsert=upsert)
        logger.info(f"MONGO: Updated: {str(oid)}")
        logger.info(f"MONGO: Matches: {result.matched_count}")
        logger.info(f"MONGO: Modified: {result.modified_count}")
        return result
    except Exception as exc:
        raise ValueError("Unable to update the document by id") from exc


def find_one(col: Collection, query: dict) -> Optional[Any]:
    try:
        res = col.find_one(query)
        if res is None:
            logger.info("MONGO: Document not found")
        else:
            logger.info(f"MONGO: Get: {res['_id']}")
        return res
    except Exception as exc:
        raise ValueError("Unable to find the document") from exc

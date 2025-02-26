# src/rabbitmq.py
import os
import aio_pika
from pylo import get_logger

# https://tenacity.readthedocs.io/en/latest/
from tenacity import retry, stop_after_attempt, wait_fixed


logger = get_logger()

RABBITMQ_URI = os.getenv("RABBITMQ_URI")
RABBITMQ_QUEUE = os.getenv("RABBITMQ_QUEUE")


@retry(wait=wait_fixed(10), stop=stop_after_attempt(10))
async def listen() -> None:
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust(url=RABBITMQ_URI)
    logger.info(f"Connected to: {RABBITMQ_URI}")

    # Create a channel
    async with connection.channel() as channel:
        # Declare the queue
        queue = await channel.declare_queue(RABBITMQ_QUEUE, durable=True)
        logger.info(f"Declared queue: {RABBITMQ_QUEUE}")

        logger.info("Waiting for messages...")
        # Start listening to the queue
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    logger.info(f"Received message: {message.body.decode()}")
                    # Perform a coroutine here...

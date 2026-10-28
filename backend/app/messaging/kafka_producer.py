"""Event bus abstraction + implementations.

publish() is thread-safe, NON-BLOCKING and returns immediately: it enqueues into a bounded buffer that a
background asyncio task drains to Kafka. Backpressure policy: when the buffer is full (broker slow/down)
the NEW message is dropped and counted in kafka_publish_errors_total{reason="buffer_full"}; inference never waits
on the network. Control messages (commands) are published through `publish_critical`, which uses a
separate, larger-reserved bucket and awaits delivery."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Awaitable, Callable

from app.core.config import Settings
from app.monitoring import metrics

log = logging.getLogger(__name__)
Handler = Callable[[str, str | None, dict], Awaitable[None]]


class EventBus:
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def publish(self, topic: str, key: str | None, value: dict) -> bool: ...
    async def publish_critical(self, topic: str, key: str | None, value: dict) -> None: ...
    async def subscribe(self, topics: list[str], group: str, handler: Handler, from_beginning: bool = False) -> None: ...
    def healthy(self) -> bool: return True


class MemoryBus(EventBus):
    """In-process bus for tests / single-process dev. Same semantics (bounded, non-blocking publish)."""

    def __init__(self, buffer_size: int = 5000):
        self._subs: list[tuple[set[str], Handler]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._q: asyncio.Queue | None = None
        self._size = buffer_size
        self._task: asyncio.Task | None = None
        self.published: list[tuple[str, str | None, dict]] = []  # inspection in tests (bounded)
        self._lock = threading.Lock()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._q = asyncio.Queue(self._size)
        self._task = asyncio.create_task(self._pump())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _pump(self) -> None:
        assert self._q
        while True:
            topic, key, value = await self._q.get()
            for topics, h in list(self._subs):
                if topic in topics:
                    try:
                        await h(topic, key, value)
                    except Exception:  # noqa: BLE001
                        log.exception("memory bus handler failed")

    def _put(self, item) -> bool:
        assert self._q
        try:
            self._q.put_nowait(item)
            return True
        except asyncio.QueueFull:
            metrics.kafka_publish_errors_total.labels("memory", "buffer_full").inc()
            return False

    def publish(self, topic: str, key: str | None, value: dict) -> bool:
        if not self._loop:
            return False
        with self._lock:
            self.published.append((topic, key, value))
            del self.published[:-2000]
        self._loop.call_soon_threadsafe(self._put, (topic, key, value))
        return True

    async def publish_critical(self, topic, key, value) -> None:
        with self._lock:
            self.published.append((topic, key, value))
        assert self._q
        await self._q.put((topic, key, value))

    async def subscribe(self, topics, group, handler, from_beginning=False) -> None:
        self._subs.append((set(topics), handler))


class KafkaBus(EventBus):
    def __init__(self, s: Settings, client_suffix: str = ""):
        self.s = s
        self._client_id = f"{s.kafka_client_id}{client_suffix}"
        self._producer = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._q: asyncio.Queue | None = None
        self._pending = 0
        self._tasks: list[asyncio.Task] = []
        self._consumers: list = []
        self._connected = False
        self._lock = threading.Lock()

    def healthy(self) -> bool:
        return self._connected

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._q = asyncio.Queue()  # bounded manually via _pending counter (thread-safe admission)
        self._tasks.append(asyncio.create_task(self._connect_and_send()))

    async def _connect_and_send(self) -> None:
        from aiokafka import AIOKafkaProducer

        delay = 1.0
        while True:  # broker outage must not crash the pipeline: keep retrying in background
            try:
                self._producer = AIOKafkaProducer(
                    bootstrap_servers=self.s.kafka_bootstrap, client_id=self._client_id,
                    acks=self.s.kafka_acks if self.s.kafka_acks == "all" else int(self.s.kafka_acks),
                    enable_idempotence=True, request_timeout_ms=self.s.kafka_delivery_timeout_ms,
                    linger_ms=20, value_serializer=lambda v: json.dumps(v, separators=(",", ":")).encode(),
                    key_serializer=lambda k: k.encode() if k else None)
                await self._producer.start()
                self._connected = True
                log.info("kafka producer connected to %s", self.s.kafka_bootstrap)
                break
            except Exception as e:  # noqa: BLE001
                self._connected = False
                metrics.kafka_publish_errors_total.labels("producer", "connect_failed").inc()
                log.warning("kafka producer connect failed (%s); retry in %.0fs", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)
        assert self._q
        while True:
            topic, key, value, fut = await self._q.get()
            try:
                await self._send_with_retry(topic, key, value)
                if fut and not fut.done():
                    fut.set_result(None)
            except Exception as e:  # noqa: BLE001
                if fut and not fut.done():
                    fut.set_exception(e)
            finally:
                with self._lock:
                    self._pending -= 1

    async def _send_with_retry(self, topic, key, value) -> None:
        last: Exception | None = None
        for attempt in range(self.s.kafka_retries):
            try:
                await self._producer.send_and_wait(topic, value=value, key=key)
                return
            except Exception as e:  # noqa: BLE001
                last = e
                metrics.kafka_publish_errors_total.labels("producer", "send_failed").inc()
                await asyncio.sleep(min(0.2 * 2 ** attempt, 5))
        raise last or RuntimeError("send failed")

    def publish(self, topic: str, key: str | None, value: dict) -> bool:
        if not self._loop or not self._q:
            return False
        with self._lock:
            if self._pending >= self.s.kafka_buffer_size:
                metrics.kafka_publish_errors_total.labels("producer", "buffer_full").inc()
                return False
            self._pending += 1
        self._loop.call_soon_threadsafe(self._q.put_nowait, (topic, key, value, None))
        return True

    async def publish_critical(self, topic, key, value) -> None:
        assert self._q
        fut = asyncio.get_running_loop().create_future()
        with self._lock:
            self._pending += 1  # critical messages bypass the buffer limit
        await self._q.put((topic, key, value, fut))
        await asyncio.wait_for(fut, timeout=self.s.kafka_delivery_timeout_ms / 1000)

    async def subscribe(self, topics, group, handler, from_beginning=False) -> None:
        self._tasks.append(asyncio.create_task(self._consume(topics, group, handler, from_beginning)))

    async def _consume(self, topics, group, handler, from_beginning) -> None:
        from aiokafka import AIOKafkaConsumer

        delay = 1.0
        while True:
            consumer = AIOKafkaConsumer(
                *topics, bootstrap_servers=self.s.kafka_bootstrap, group_id=group, client_id=self._client_id,
                enable_auto_commit=False, auto_offset_reset="earliest" if from_beginning else "latest",
                value_deserializer=lambda b: json.loads(b.decode()), key_deserializer=lambda b: b.decode() if b else None)
            try:
                await consumer.start()
                self._consumers.append(consumer)
                delay = 1.0
                async for msg in consumer:
                    try:
                        await handler(msg.topic, msg.key, msg.value)
                    except Exception:  # noqa: BLE001 - poison message must not wedge the consumer
                        log.exception("handler failed for %s offset %s", msg.topic, msg.offset)
                    await consumer.commit()  # at-least-once; handlers must be idempotent
            except asyncio.CancelledError:
                await consumer.stop()
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("kafka consumer error (%s); retry in %.0fs", e, delay)
                try:
                    await consumer.stop()
                except Exception:  # noqa: BLE001
                    pass
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self._producer:
            try:
                await asyncio.wait_for(self._producer.stop(), 5)
            except Exception:  # noqa: BLE001
                pass


def make_bus(s: Settings, suffix: str = "") -> EventBus:
    return MemoryBus(s.kafka_buffer_size) if s.bus == "memory" else KafkaBus(s, suffix)


async def ensure_topics(s: Settings) -> None:
    """Create topics idempotently (partitions/retention from topics.TOPIC_CONFIG)."""
    from aiokafka.admin import AIOKafkaAdminClient, NewTopic
    from aiokafka.errors import TopicAlreadyExistsError

    from app.messaging.topics import TOPIC_CONFIG

    admin = AIOKafkaAdminClient(bootstrap_servers=s.kafka_bootstrap, client_id=s.kafka_client_id + "-admin")
    await admin.start()
    try:
        for name, cfg in TOPIC_CONFIG.items():
            try:
                await admin.create_topics([NewTopic(name, cfg["partitions"] or s.kafka_topic_partitions,
                                                    s.kafka_replication, topic_configs={"retention.ms": str(cfg["retention_ms"])})])
            except TopicAlreadyExistsError:
                pass
    finally:
        await admin.close()

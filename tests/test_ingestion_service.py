from src.ingestion_service import InMemoryIngestionQueue, IngestionService


def test_enqueue_and_process_one():
    queue = InMemoryIngestionQueue()
    service = IngestionService(queue)

    service.enqueue_record({"duration": 1, "src_bytes": 20, "dst_bytes": 5}, source="unit")
    assert queue.size() == 1

    output = service.process_one(lambda payload, source: {"source": source, "duration": payload["duration"]})
    assert output is not None
    assert output["source"] == "unit"
    assert output["result"]["duration"] == 1
    assert queue.size() == 0


def test_invalid_numeric_value_raises():
    queue = InMemoryIngestionQueue()
    service = IngestionService(queue)

    try:
        service.enqueue_record({"duration": "not-a-number"})
        assert False, "Expected ValueError"
    except ValueError:
        assert True

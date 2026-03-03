def handler(event, context):
    from worker import ChunkWorker
    worker = ChunkWorker()
    processed = worker.run_once()
    return {"processed": processed}
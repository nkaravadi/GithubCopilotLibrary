"""Live test: embeddings through CopilotClient (shared session token)."""
import sys, time
sys.path.insert(0, 'src')
from copilotlibrary import CopilotClient

with CopilotClient() as client:
    # First call — session token exchange happens here
    t0 = time.perf_counter()
    v1 = client.get_embedding("Binary search is O(log n)")
    t1 = time.perf_counter()
    print(f"1st call: {len(v1)} dims  ({(t1-t0)*1000:.0f} ms — includes token exchange)")

    # Second call — exchange is cached, just the REST call
    t2 = time.perf_counter()
    v2 = client.get_embedding("Hash tables give O(1) lookups")
    t3 = time.perf_counter()
    print(f"2nd call: {len(v2)} dims  ({(t3-t2)*1000:.0f} ms — cached token)")

    # Batch — single REST call for all texts
    texts = ["Python is interpreted", "Rust is compiled", "Go has goroutines"]
    t4 = time.perf_counter()
    vecs = client.get_embeddings_batch(texts)
    t5 = time.perf_counter()
    print(f"Batch  : {len(vecs)} vectors x {len(vecs[0])} dims  ({(t5-t4)*1000:.0f} ms)")

    # Confirm chat still works in the same session
    resp = client.chat("Say 'ok' and nothing else")
    print(f"Chat   : {resp.content.strip()!r}")

    print(f"\n✓ Same CopilotClient instance used for both chat and embeddings")


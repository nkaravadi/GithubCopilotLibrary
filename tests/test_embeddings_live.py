"""Live test of embedding auto-discovery."""
import sys
sys.path.insert(0, 'src')
from copilotlibrary.embeddings import CopilotEmbeddings, find_copilot_token

# First confirm token is found
token = find_copilot_token()
if token:
    print(f"Token auto-discovered: {token[:8]}...{token[-4:]} (masked)")
else:
    print("No token found — sign in to Copilot in IntelliJ first")
    sys.exit(1)

# Now test embeddings
print("\nTesting embedding API...")
try:
    emb = CopilotEmbeddings()
    result = emb.embed("Binary search is an efficient algorithm.")
    print(f"✓ Success!")
    print(f"  Model:      {result.model}")
    print(f"  Dimensions: {result.dimensions}")
    print(f"  First 5:    {[round(v,4) for v in result.vector[:5]]}")
    print(f"  Usage:      {result.usage}")

    # Batch test
    print("\nBatch embedding test...")
    results = emb.embed_batch(["Hello world", "Binary search", "Python is great"])
    print(f"✓ Got {len(results)} embeddings")
    for r in results:
        print(f"  '{r.input_text[:20]}' → {r.dimensions} dims, first val: {round(r.vector[0],4)}")

    # Different model
    print("\nLarge model test...")
    result_large = emb.embed("Hello world", model="text-embedding-3-large")
    print(f"✓ text-embedding-3-large: {result_large.dimensions} dims")

except RuntimeError as e:
    print(f"✗ API error: {e}")


"""
Minimal example: call the evaleval entity-registry API.
"""
import requests

# --- health check ---
r = requests.get("https://evaleval-entity-registry.hf.space/api/v1/health", timeout=10)
print("health:", r.json())

# --- resolve one entity ---
r = requests.post(
    "https://evaleval-entity-registry.hf.space/api/v1/resolve",
    json={"raw_value": "MATH Level 5", "entity_type": "benchmark"},
    timeout=10,
)
print("\nresolve 'MATH Level 5':", r.json())

# --- resolve with no match (returns canonical_id=null) ---
r = requests.post(
    "https://evaleval-entity-registry.hf.space/api/v1/resolve",
    json={"raw_value": "TotallyFakeBenchmark", "entity_type": "benchmark"},
    timeout=10,
)
print("\nresolve 'TotallyFakeBenchmark':", r.json())

# --- batch resolve ---
r = requests.post(
    "https://evaleval-entity-registry.hf.space/api/v1/resolve/batch",
    json=[
        {"raw_value": "MATH", "entity_type": "benchmark"},
        {"raw_value": "Exact Match", "entity_type": "metric"},
        {"raw_value": "LM Evaluation Harness", "entity_type": "harness"},
    ],
    timeout=30,
)
print("\nbatch resolve:", r.json())

# --- list benchmarks (GET) ---
r = requests.get(
    "https://evaleval-entity-registry.hf.space/api/v1/benchmarks",
    params={"search": "math"},
    timeout=10,
)
print("\nbenchmarks matching 'math':", [b["id"] for b in r.json()])

# --- look up a single entity by canonical id ---
r = requests.get(
    "https://evaleval-entity-registry.hf.space/api/v1/benchmarks/math",
    timeout=10,
)
print("\nbenchmark 'math':", r.json())

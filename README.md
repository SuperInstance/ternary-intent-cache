# ternary-intent-cache

**Cache the compiler. When the same intent produces the same bytecode, don't recompile.**

In the Oxide Stack, the "pincher" layer compiles natural-language intents into Flux bytecode. "Rotate the tensor 90° clockwise" and "Turn the matrix right a quarter turn" mean the same thing and should produce identical bytecode. This cache sits between the intent parser and the bytecode compiler, returning cached results on hit and compiling + storing on miss.

## The Insight

LLM-as-compiler is expensive. Every intent compilation burns tokens and takes milliseconds to seconds. But users repeat themselves — the same queries come back in loops, slightly rephrased. If you hash the intent string and cache the resulting bytecode, repeated queries become O(1) lookups.

The tricky part is *when* to evict. LRU is the standard answer, but in a GPU inference pipeline, you also care about TTL — old bytecodes may reference stale model versions. This crate implements LRU eviction with a monotonic logical clock (each `lookup` and `store` ticks the clock). No wall-clock TTL yet, but the `last_used` timestamp makes it straightforward to add.

## Quick Start

```toml
[dependencies]
ternary-intent-cache = "0.1.0"
```

```rust
use ternary_intent_cache::*;

let mut cache = IntentCache::new(256); // max 256 entries

// First call: miss → compile → store
let ops = cache.compile_or_cache("add two ternary vectors", |intent| {
    // Your compilation logic here
    vec![BytecodeOp::TAdd, BytecodeOp::Halt]
});
assert_eq!(cache.misses(), 1);

// Second call: hit → return cached
let ops2 = cache.compile_or_cache("add two ternary vectors", |intent| {
    panic!("should not be called — cache hit!");
});
assert_eq!(ops, ops2);
assert_eq!(cache.hits(), 1);

// Check hit rate
println!("Hit rate: {:.1}%", cache.hit_rate() * 100.0);
```

## Architecture

```
         Intent String
              │
              ▼
     ┌────────────────┐
     │  hash_intent()  │  DJB2 hash → u64
     └────────┬───────┘
              │
              ▼
     ┌────────────────────┐
     │  HashMap<u64,      │
     │    CachedProgram>  │
     │                    │
     │  ┌──────────────┐  │
     │  │ intent_hash  │  │
     │  │ ops: Vec<Op> │  │
     │  │ created_at   │  │
     │  │ last_used    │  │  ← LRU eviction target
     │  │ hits: u64    │  │
     │  └──────────────┘  │
     └────────────────────┘
              │
       miss?  │  hit → return ops
              ▼
     compile_fn(intent)
              │
              ▼
         store result
              │
              ▼
         evict LRU if at capacity
```

The cache is a `HashMap<u64, CachedProgram>` keyed by a DJB2 hash of the intent string. Each `CachedProgram` tracks when it was created, when it was last accessed, and how many times it's been hit. When the cache is full, `store` evicts the entry with the oldest `last_used` timestamp.

## API Reference

### BytecodeOp

```rust
pub enum BytecodeOp {
    TAdd,   // Ternary addition
    TMul,   // Ternary multiplication
    TNeg,   // Ternary negation
    Sync,   // Synchronization barrier
    Halt,   // Program termination
}
```

Placeholder bytecode operations. In production, these would be the full Flux VM instruction set.

### IntentCache

```rust
IntentCache::new(max_size: usize) -> IntentCache
cache.lookup(intent: &str) -> Option<Vec<BytecodeOp>>
cache.store(intent: &str, ops: Vec<BytecodeOp>)
cache.compile_or_cache(intent: &str, compile_fn: impl Fn(&str) -> Vec<BytecodeOp>) -> Vec<BytecodeOp>
cache.invalidate(intent: &str) -> bool
cache.hit_rate() -> f64
cache.size() -> usize
cache.hits() -> u64
cache.misses() -> u64
```

- **`new`** — create cache with maximum entry count
- **`lookup`** — check cache. On hit, updates `last_used` and increments hit counters. On miss, increments miss counter and returns `None`.
- **`store`** — insert compiled bytecode. Evicts LRU entry if at capacity.
- **`compile_or_cache`** — the workhorse. Tries lookup; on miss, calls `compile_fn`, stores the result, and returns it.
- **`invalidate`** — remove a specific entry. Returns `true` if it existed.
- **`hit_rate`** — `hits / (hits + misses)`, or 0.0 if no accesses yet.

### CachedProgram

```rust
pub struct CachedProgram {
    pub intent_hash: u64,
    pub ops: Vec<BytecodeOp>,
    pub created_at: u64,   // logical clock tick when stored
    pub last_used: u64,    // logical clock tick of last access
    pub hits: u64,         // number of cache hits
}
```

## Real-World Example: Inference Server

```rust
use ternary_intent_cache::*;

// Shared cache across all inference threads
let mut cache = IntentCache::new(4096);

// Simulate an inference server receiving queries
let queries = vec![
    "classify this image as cat or dog",
    "translate to french: hello world",
    "classify this image as cat or dog", // repeat → cache hit
    "summarize this document",
    "classify this image as cat or dog", // repeat → cache hit
];

for query in &queries {
    let bytecode = cache.compile_or_cache(query, |intent| {
        // In production: call LLM to compile intent → Flux bytecode
        compile_intent_to_bytecode(intent)
    });
    execute(bytecode);
}

println!("Hit rate: {:.1}%", cache.hit_rate() * 100.0);
// With 3 repeats out of 5 queries: 40% hit rate

// Monitor cache health
println!("Cache size: {}/{}", cache.size(), 4096);
println!("Total hits: {}", cache.hits());
println!("Total misses: {}", cache.misses());
```

## LRU Eviction

When `store` is called and the cache is full, it scans all entries to find the one with the smallest `last_used` value and removes it. This is O(n) per eviction — fine for small caches (<10K entries), but for larger ones you'd want a proper LRU data structure (doubly-linked list + hashmap).

The tradeoff is intentional: simplicity over performance. This cache sits in front of an LLM compilation step that takes milliseconds to seconds. An O(n) eviction scan is noise.

## Hashing

`hash_intent` uses DJB2 (Daniel J. Bernstein's hash function): simple, fast, and good enough for cache keys. It's *not* collision-resistant in the cryptographic sense — two different intents can hash to the same u64. In practice, this is fine for a cache (the worst case is a false positive hit, returning slightly wrong bytecode). For production use, consider SHA-256 or xxHash.

## Ecosystem

- **ternary-fault-tree** — analyze cache reliability under GPU failure
- **ternary-antidote** — distribute cache state across nodes with CRDTs
- **ternary-shard** — partition cached bytecodes across GPU nodes

## Open Questions

- **Semantic caching**: DJB2 hashes exact strings. "Add two vectors" and "Sum two vectors" hash differently but mean the same thing. Embedding-based similarity would catch semantic duplicates.
- **Wall-clock TTL**: The `created_at` field uses a logical clock. Adding a wall-clock timestamp would enable time-based expiry for stale compilations.
- **Concurrent access**: The current implementation is single-threaded. A concurrent version would need either a `Mutex<HashMap>` or a lock-free concurrent cache.
- **Bytecode validation**: No mechanism to check if cached bytecode is still valid for the current model version.

## Stats

| Metric | Value |
|--------|-------|
| Tests | 6 |
| Lines of Rust | 151 |
| Public API | 12 items |

## License

Apache-2.0

//! # ternary-intent-cache
//!
//! Cache for intent→bytecode compilations.
//! Same intent = cached Flux bytecode. LRU + TTL.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BytecodeOp { TAdd, TMul, TNeg, Sync, Halt }

#[derive(Debug, Clone)]
pub struct CachedProgram {
    pub intent_hash: u64,
    pub ops: Vec<BytecodeOp>,
    pub created_at: u64,
    pub last_used: u64,
    pub hits: u64,
}

pub struct IntentCache {
    cache: HashMap<u64, CachedProgram>,
    max_size: usize,
    time: u64,
    hits: u64,
    misses: u64,
}

fn hash_intent(intent: &str) -> u64 {
    let mut h: u64 = 5381;
    for b in intent.bytes() { h = h.wrapping_mul(33).wrapping_add(b as u64); }
    h
}

impl IntentCache {
    pub fn new(max_size: usize) -> Self {
        Self { cache: HashMap::new(), max_size, time: 0, hits: 0, misses: 0 }
    }

    fn tick(&mut self) { self.time += 1; }

    pub fn lookup(&mut self, intent: &str) -> Option<Vec<BytecodeOp>> {
        self.tick();
        let h = hash_intent(intent);
        if let Some(prog) = self.cache.get_mut(&h) {
            prog.last_used = self.time;
            prog.hits += 1;
            self.hits += 1;
            Some(prog.ops.clone())
        } else { self.misses += 1; None }
    }

    pub fn store(&mut self, intent: &str, ops: Vec<BytecodeOp>) {
        self.tick();
        if self.cache.len() >= self.max_size {
            // Evict LRU
            if let Some(lru_key) = self.cache.values().min_by_key(|p| p.last_used).map(|p| p.intent_hash) {
                self.cache.remove(&lru_key);
            }
        }
        let h = hash_intent(intent);
        self.cache.insert(h, CachedProgram { intent_hash: h, ops, created_at: self.time, last_used: self.time, hits: 0 });
    }

    pub fn compile_or_cache(&mut self, intent: &str, compile_fn: impl Fn(&str) -> Vec<BytecodeOp>) -> Vec<BytecodeOp> {
        if let Some(ops) = self.lookup(intent) { return ops; }
        let ops = compile_fn(intent);
        self.store(intent, ops.clone());
        ops
    }

    pub fn invalidate(&mut self, intent: &str) -> bool {
        let h = hash_intent(intent);
        self.cache.remove(&h).is_some()
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }

    pub fn size(&self) -> usize { self.cache.len() }
    pub fn hits(&self) -> u64 { self.hits }
    pub fn misses(&self) -> u64 { self.misses }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_compile(s: &str) -> Vec<BytecodeOp> {
        if s.contains("add") { vec![BytecodeOp::TAdd, BytecodeOp::Halt] }
        else { vec![BytecodeOp::TNeg, BytecodeOp::Halt] }
    }

    #[test]
    fn test_miss_then_hit() {
        let mut c = IntentCache::new(10);
        assert!(c.lookup("add stuff").is_none());
        c.store("add stuff", mock_compile("add"));
        assert!(c.lookup("add stuff").is_some());
        assert_eq!(c.hits(), 1);
        assert_eq!(c.misses(), 1);
    }

    #[test]
    fn test_compile_or_cache() {
        let mut c = IntentCache::new(10);
        let ops1 = c.compile_or_cache("add", mock_compile);
        assert!(c.misses() > 0);
        let ops2 = c.compile_or_cache("add", mock_compile);
        assert_eq!(ops1, ops2);
        assert!(c.hits() > 0);
    }

    #[test]
    fn test_lru_eviction() {
        let mut c = IntentCache::new(2);
        c.store("a", vec![BytecodeOp::Halt]);
        c.store("b", vec![BytecodeOp::Halt]);
        c.store("c", vec![BytecodeOp::Halt]); // evicts "a"
        assert_eq!(c.size(), 2);
        assert!(c.lookup("a").is_none());
        assert!(c.lookup("b").is_some());
    }

    #[test]
    fn test_invalidation() {
        let mut c = IntentCache::new(10);
        c.store("target", vec![BytecodeOp::Halt]);
        assert!(c.invalidate("target"));
        assert!(c.lookup("target").is_none());
    }

    #[test]
    fn test_hit_rate() {
        let mut c = IntentCache::new(10);
        c.compile_or_cache("a", mock_compile); // miss
        c.compile_or_cache("a", mock_compile); // hit
        c.compile_or_cache("a", mock_compile); // hit
        assert!((c.hit_rate() - 0.667).abs() < 0.05);
    }

    #[test]
    fn test_different_intents() {
        let mut c = IntentCache::new(10);
        let ops1 = c.compile_or_cache("add things", mock_compile);
        let ops2 = c.compile_or_cache("negate things", mock_compile);
        assert_ne!(ops1, ops2);
    }
}

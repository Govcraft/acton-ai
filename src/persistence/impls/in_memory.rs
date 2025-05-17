use std::collections::HashMap;
use dashmap::DashMap;
use mti::prelude::MagicTypeId;
use crate::agents::common::Conversation;
use crate::persistence::memory_store::MemoryStore;

#[derive(Default, Clone)]
pub struct InMemory {
    memory: DashMap<MagicTypeId, Conversation>
}
impl MemoryStore for InMemory {
    fn get_all(&self) -> Vec<Conversation> {
        self.memory.iter().map(|entry| entry.value().clone()).collect()
    }

    fn get_by_id(&self, id: &MagicTypeId) -> Option<Conversation> {
        self.memory.get(id).map(|value| value.clone())
    }

    fn store(&mut self, conversation: Conversation) {
        println!("Memory stored:{}", conversation.id());
         self.memory.insert(conversation.id().clone(), conversation);
    }
}
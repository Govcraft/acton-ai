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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_and_retrieve_conversation() {
        let mut store = InMemory::default();
        let conv = Conversation::default();
        store.store(conv.clone());

        assert_eq!(store.get_all().len(), 1);
        assert!(store.get_by_id(conv.id()).is_some());
    }
}

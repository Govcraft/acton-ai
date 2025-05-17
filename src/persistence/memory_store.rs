use mti::prelude::MagicTypeId;
use crate::agents::common::Conversation;

pub trait MemoryStore {
    fn get_all(&self) -> Vec<Conversation>;
    fn get_by_id(&self, id: &MagicTypeId) -> Option<Conversation>;
    fn store(&self, conversation: Conversation);
}
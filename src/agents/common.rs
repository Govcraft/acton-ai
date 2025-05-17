use mti::prelude::*;
use crate::messages::*;

#[derive(Debug)]
#[derive(Clone)]
#[derive(Eq, Hash, PartialEq)]
pub struct Conversation(MagicTypeId);
impl Conversation {
    pub fn new() -> Self {
        Self( "conversation".create_type_id::<V7>() )
    }
    pub fn id(&self) -> &MagicTypeId {
        &self.0
    }   
}
impl Default for Conversation {
    fn default() -> Self {
        Self::new()
    }   
}

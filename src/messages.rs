use acton_reactive::prelude::*;
use mti::prelude::MagicTypeId;
use crate::agents::common::Conversation;

#[acton_message]
pub struct UserQueryReceivedEvent {
    pub query: String,
    pub user_id: String,
    pub conversation_id: MagicTypeId,
}

#[acton_message]
pub struct AiAgentTaskStartedEvent {
    pub agent_id: Ern,
    pub conversation_id: MagicTypeId,
    pub goal: String,
}

#[acton_message]
pub struct ModelQueryRequestMsg {
    pub prompt: String,
    pub history: Conversation,
}

#[acton_message]
pub struct ModelResponseMsg {
    pub response_text: String,
    // ... additional fields as specified
    pub conversation_id: MagicTypeId,
}

#[acton_message]
pub struct ModelResponseGeneratedEvent {
    pub conversation_id: MagicTypeId,
    pub response_id: MagicTypeId,
    pub model_id: String,
}

#[acton_message]
pub struct ToolExecutionRequestMsg {
    pub tool_name: String,
    // The type for 'arguments' was '...' in the specification.
    // Using String as a placeholder. Consider using a more specific type
    // like serde_json::Value or a custom struct/enum if appropriate.
    pub arguments: String,
}

#[acton_message]
pub struct ToolResultMsg {
    pub tool_name: String,
    pub result: String,
    // ... additional fields as specified at the end
}

#[acton_message]
pub struct ToolExecutedEvent {
    pub conversation_id: MagicTypeId,
    pub tool_name: String,
    pub status: String,
}

#[acton_message]
pub struct ToolAvailableMsg {
    pub name: String,
    pub description: String,
}

#[acton_message]
pub struct MemoryStoreRequestMsg {
    pub conversation: Conversation,
    // pub turn: InteractionTurn,
}

#[acton_message]
pub struct MemoryRetrieveRequestMsg {
    pub conversation: Conversation,
    // ... additional fields as specified
}

#[acton_message]
pub struct MemoryRetrievedMsg {
    pub history: Vec<String>,
    // ... additional fields as specified at the end
}

#[acton_message]
pub struct ConversationUpdatedEvent {
    pub conversation_id: MagicTypeId,
    pub last_turn_summary: String,
}

#[acton_message]
pub struct ValidateContentRequestMsg {
    pub content: String,
    // ... additional fields as specified
}

#[acton_message]
pub struct GuardrailValidationResultMsg {
    pub is_safe: bool,
    // ... additional fields as specified at the end
}

#[acton_message]
pub struct SecurityAlertEvent {
    // pub severity: AlertSeverity,
    pub details: String,
    pub offending_content: Option<String>,
}

#[acton_message]
pub struct HumanInterventionRequiredEvent {
    pub conversation_id: MagicTypeId,
    pub prompt_for_human: String,
    pub context_summary: String,
    pub hil_request_id: MagicTypeId,
}

#[acton_message]
pub struct HumanInputProvidedMsg {
    pub hil_request_id: MagicTypeId,
    pub input: String,
    pub decision: String,
}

#[acton_message]
pub struct HumanInputReceivedEvent {
    pub conversation_id: MagicTypeId,
    pub hil_request_id: MagicTypeId,
}

#[acton_message]
pub struct FinalAnswerGeneratedEvent {
    pub conversation_id: MagicTypeId,
    pub answer: String,
}
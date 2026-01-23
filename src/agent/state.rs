//! Agent state enumeration.
//!
//! Defines the possible states an agent can be in during its lifecycle.

use serde::{Deserialize, Serialize};
use std::fmt;

/// The current state of an agent in its reasoning loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum AgentState {
    /// Agent is idle, waiting for input
    #[default]
    Idle,
    /// Agent is processing and reasoning about the input
    Thinking,
    /// Agent is executing a tool
    Executing,
    /// Agent is waiting for external input (tool result, user confirmation)
    Waiting,
    /// Agent has completed its task
    Completed,
    /// Agent is stopping
    Stopping,
}

impl AgentState {
    /// Returns true if the agent can accept new user prompts in this state.
    #[must_use]
    pub fn can_accept_prompt(&self) -> bool {
        matches!(self, Self::Idle | Self::Completed)
    }

    /// Returns true if the agent is actively processing.
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Thinking | Self::Executing | Self::Waiting)
    }

    /// Returns true if the agent is in a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Stopping)
    }
}

impl fmt::Display for AgentState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "idle"),
            Self::Thinking => write!(f, "thinking"),
            Self::Executing => write!(f, "executing"),
            Self::Waiting => write!(f, "waiting"),
            Self::Completed => write!(f, "completed"),
            Self::Stopping => write!(f, "stopping"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_is_idle() {
        assert_eq!(AgentState::default(), AgentState::Idle);
    }

    #[test]
    fn can_accept_prompt_when_idle_or_completed() {
        assert!(AgentState::Idle.can_accept_prompt());
        assert!(AgentState::Completed.can_accept_prompt());
        assert!(!AgentState::Thinking.can_accept_prompt());
        assert!(!AgentState::Executing.can_accept_prompt());
        assert!(!AgentState::Waiting.can_accept_prompt());
        assert!(!AgentState::Stopping.can_accept_prompt());
    }

    #[test]
    fn is_active_when_processing() {
        assert!(!AgentState::Idle.is_active());
        assert!(AgentState::Thinking.is_active());
        assert!(AgentState::Executing.is_active());
        assert!(AgentState::Waiting.is_active());
        assert!(!AgentState::Completed.is_active());
        assert!(!AgentState::Stopping.is_active());
    }

    #[test]
    fn is_terminal_only_when_stopping() {
        assert!(!AgentState::Idle.is_terminal());
        assert!(!AgentState::Thinking.is_terminal());
        assert!(!AgentState::Completed.is_terminal());
        assert!(AgentState::Stopping.is_terminal());
    }

    #[test]
    fn display_format() {
        assert_eq!(AgentState::Idle.to_string(), "idle");
        assert_eq!(AgentState::Thinking.to_string(), "thinking");
        assert_eq!(AgentState::Executing.to_string(), "executing");
        assert_eq!(AgentState::Waiting.to_string(), "waiting");
        assert_eq!(AgentState::Completed.to_string(), "completed");
        assert_eq!(AgentState::Stopping.to_string(), "stopping");
    }

    #[test]
    fn serialization_roundtrip() {
        let states = vec![
            AgentState::Idle,
            AgentState::Thinking,
            AgentState::Executing,
            AgentState::Waiting,
            AgentState::Completed,
            AgentState::Stopping,
        ];

        for state in states {
            let json = serde_json::to_string(&state).unwrap();
            let deserialized: AgentState = serde_json::from_str(&json).unwrap();
            assert_eq!(state, deserialized);
        }
    }
}

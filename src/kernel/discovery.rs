//! Agent capability registry for discovery.
//!
//! This module provides the `CapabilityRegistry` which tracks which agents
//! have which capabilities, enabling agent discovery by capability.

use crate::types::AgentId;
use std::collections::{HashMap, HashSet};

/// Registry for tracking agent capabilities.
///
/// This is a pure data structure used by the Kernel to track which agents
/// have announced which capabilities, enabling agent discovery.
#[derive(Debug, Clone, Default)]
pub struct CapabilityRegistry {
    /// Maps capability name to set of agent IDs that have that capability
    capability_to_agents: HashMap<String, HashSet<AgentId>>,
    /// Maps agent ID to set of capabilities that agent has
    agent_to_capabilities: HashMap<AgentId, HashSet<String>>,
}

impl CapabilityRegistry {
    /// Creates a new empty capability registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            capability_to_agents: HashMap::new(),
            agent_to_capabilities: HashMap::new(),
        }
    }

    /// Registers capabilities for an agent.
    ///
    /// This replaces any existing capabilities for the agent.
    pub fn register(&mut self, agent_id: AgentId, capabilities: Vec<String>) {
        // Remove old capabilities
        self.unregister(&agent_id);

        // Add new capabilities
        for cap in &capabilities {
            self.capability_to_agents
                .entry(cap.clone())
                .or_default()
                .insert(agent_id.clone());
        }

        self.agent_to_capabilities
            .insert(agent_id, capabilities.into_iter().collect());
    }

    /// Unregisters all capabilities for an agent.
    pub fn unregister(&mut self, agent_id: &AgentId) {
        if let Some(caps) = self.agent_to_capabilities.remove(agent_id) {
            for cap in caps {
                if let Some(agents) = self.capability_to_agents.get_mut(&cap) {
                    agents.remove(agent_id);
                    if agents.is_empty() {
                        self.capability_to_agents.remove(&cap);
                    }
                }
            }
        }
    }

    /// Finds an agent with the specified capability.
    ///
    /// Returns the first agent found, or None if no agent has the capability.
    #[must_use]
    pub fn find_capable_agent(&self, capability: &str) -> Option<AgentId> {
        self.capability_to_agents
            .get(capability)
            .and_then(|agents| agents.iter().next().cloned())
    }

    /// Finds all agents with the specified capability.
    #[must_use]
    pub fn find_all_capable_agents(&self, capability: &str) -> Vec<AgentId> {
        self.capability_to_agents
            .get(capability)
            .map(|agents| agents.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Gets all capabilities registered for an agent.
    #[must_use]
    pub fn get_agent_capabilities(&self, agent_id: &AgentId) -> Vec<String> {
        self.agent_to_capabilities
            .get(agent_id)
            .map(|caps| caps.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Checks if an agent has a specific capability.
    #[must_use]
    pub fn has_capability(&self, agent_id: &AgentId, capability: &str) -> bool {
        self.agent_to_capabilities
            .get(agent_id)
            .map(|caps| caps.contains(capability))
            .unwrap_or(false)
    }

    /// Returns the total number of registered agents.
    #[must_use]
    pub fn agent_count(&self) -> usize {
        self.agent_to_capabilities.len()
    }

    /// Returns the total number of unique capabilities.
    #[must_use]
    pub fn capability_count(&self) -> usize {
        self.capability_to_agents.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_registry_is_empty() {
        let registry = CapabilityRegistry::new();
        assert_eq!(registry.agent_count(), 0);
        assert_eq!(registry.capability_count(), 0);
    }

    #[test]
    fn register_capabilities() {
        let mut registry = CapabilityRegistry::new();
        let agent_id = AgentId::new();

        registry.register(
            agent_id.clone(),
            vec!["code_review".to_string(), "summarize".to_string()],
        );

        assert_eq!(registry.agent_count(), 1);
        assert_eq!(registry.capability_count(), 2);
        assert!(registry.has_capability(&agent_id, "code_review"));
        assert!(registry.has_capability(&agent_id, "summarize"));
    }

    #[test]
    fn find_capable_agent() {
        let mut registry = CapabilityRegistry::new();
        let agent_id = AgentId::new();

        registry.register(agent_id.clone(), vec!["code_review".to_string()]);

        let found = registry.find_capable_agent("code_review");
        assert_eq!(found, Some(agent_id));

        let not_found = registry.find_capable_agent("unknown");
        assert_eq!(not_found, None);
    }

    #[test]
    fn find_all_capable_agents() {
        let mut registry = CapabilityRegistry::new();
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();

        registry.register(
            agent1.clone(),
            vec!["summarize".to_string(), "code_review".to_string()],
        );
        registry.register(agent2.clone(), vec!["summarize".to_string()]);

        let summarizers = registry.find_all_capable_agents("summarize");
        assert_eq!(summarizers.len(), 2);
        assert!(summarizers.contains(&agent1));
        assert!(summarizers.contains(&agent2));

        let reviewers = registry.find_all_capable_agents("code_review");
        assert_eq!(reviewers.len(), 1);
        assert!(reviewers.contains(&agent1));
    }

    #[test]
    fn unregister_agent() {
        let mut registry = CapabilityRegistry::new();
        let agent_id = AgentId::new();

        registry.register(agent_id.clone(), vec!["code_review".to_string()]);
        assert_eq!(registry.agent_count(), 1);

        registry.unregister(&agent_id);
        assert_eq!(registry.agent_count(), 0);
        assert_eq!(registry.capability_count(), 0);
        assert_eq!(registry.find_capable_agent("code_review"), None);
    }

    #[test]
    fn re_register_replaces_capabilities() {
        let mut registry = CapabilityRegistry::new();
        let agent_id = AgentId::new();

        registry.register(agent_id.clone(), vec!["old_capability".to_string()]);
        assert!(registry.has_capability(&agent_id, "old_capability"));

        registry.register(agent_id.clone(), vec!["new_capability".to_string()]);
        assert!(!registry.has_capability(&agent_id, "old_capability"));
        assert!(registry.has_capability(&agent_id, "new_capability"));
    }

    #[test]
    fn get_agent_capabilities() {
        let mut registry = CapabilityRegistry::new();
        let agent_id = AgentId::new();

        registry.register(
            agent_id.clone(),
            vec!["cap1".to_string(), "cap2".to_string()],
        );

        let caps = registry.get_agent_capabilities(&agent_id);
        assert_eq!(caps.len(), 2);
        assert!(caps.contains(&"cap1".to_string()));
        assert!(caps.contains(&"cap2".to_string()));
    }

    #[test]
    fn get_capabilities_for_unknown_agent() {
        let registry = CapabilityRegistry::new();
        let agent_id = AgentId::new();

        let caps = registry.get_agent_capabilities(&agent_id);
        assert!(caps.is_empty());
    }
}

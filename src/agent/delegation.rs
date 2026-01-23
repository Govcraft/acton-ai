//! Task delegation state tracking.
//!
//! This module provides types for tracking delegated tasks between agents,
//! including both outgoing (tasks this agent delegated to others) and
//! incoming (tasks delegated to this agent by others).

use crate::types::{AgentId, TaskId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// State of a delegated task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DelegatedTaskState {
    /// Task has been sent, awaiting acceptance
    Pending,
    /// Task was accepted by the target agent
    Accepted,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed,
}

impl std::fmt::Display for DelegatedTaskState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Accepted => write!(f, "accepted"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

/// Tracks a delegated task.
#[derive(Debug, Clone)]
pub struct DelegatedTask {
    /// The task identifier
    pub task_id: TaskId,
    /// The agent the task was delegated to
    pub delegated_to: AgentId,
    /// The type of task
    pub task_type: String,
    /// Current state of the task
    pub state: DelegatedTaskState,
    /// When the task was created
    pub created_at: Instant,
    /// Optional deadline
    pub deadline: Option<Duration>,
    /// The result if completed
    pub result: Option<serde_json::Value>,
    /// The error message if failed
    pub error: Option<String>,
}

impl DelegatedTask {
    /// Creates a new delegated task tracker.
    #[must_use]
    pub fn new(task_id: TaskId, delegated_to: AgentId, task_type: String) -> Self {
        Self {
            task_id,
            delegated_to,
            task_type,
            state: DelegatedTaskState::Pending,
            created_at: Instant::now(),
            deadline: None,
            result: None,
            error: None,
        }
    }

    /// Sets a deadline for the task.
    #[must_use]
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Marks the task as accepted.
    pub fn accept(&mut self) {
        self.state = DelegatedTaskState::Accepted;
    }

    /// Marks the task as completed with a result.
    pub fn complete(&mut self, result: serde_json::Value) {
        self.state = DelegatedTaskState::Completed;
        self.result = Some(result);
    }

    /// Marks the task as failed with an error.
    pub fn fail(&mut self, error: impl Into<String>) {
        self.state = DelegatedTaskState::Failed;
        self.error = Some(error.into());
    }

    /// Returns true if the task has exceeded its deadline.
    #[must_use]
    pub fn is_overdue(&self) -> bool {
        if let Some(deadline) = self.deadline {
            self.created_at.elapsed() > deadline
        } else {
            false
        }
    }

    /// Returns true if the task is in a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            DelegatedTaskState::Completed | DelegatedTaskState::Failed
        )
    }
}

/// Information about an incoming task.
#[derive(Debug, Clone)]
pub struct IncomingTaskInfo {
    /// The task identifier
    pub task_id: TaskId,
    /// The agent that delegated the task
    pub from: AgentId,
    /// The type of task
    pub task_type: String,
    /// When the task was received
    pub received_at: Instant,
    /// Whether the task has been accepted
    pub accepted: bool,
}

/// Tracks all delegated tasks for an agent.
#[derive(Debug, Clone, Default)]
pub struct DelegationTracker {
    /// Tasks this agent has delegated to others
    outgoing: HashMap<TaskId, DelegatedTask>,
    /// Tasks delegated to this agent by others
    incoming: HashMap<TaskId, IncomingTaskInfo>,
}

impl DelegationTracker {
    /// Creates a new delegation tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
        }
    }

    /// Tracks a task that this agent delegated to another.
    pub fn track_outgoing(&mut self, task: DelegatedTask) {
        self.outgoing.insert(task.task_id.clone(), task);
    }

    /// Gets a mutable reference to an outgoing task.
    pub fn get_outgoing_mut(&mut self, task_id: &TaskId) -> Option<&mut DelegatedTask> {
        self.outgoing.get_mut(task_id)
    }

    /// Gets an outgoing task by ID.
    #[must_use]
    pub fn get_outgoing(&self, task_id: &TaskId) -> Option<&DelegatedTask> {
        self.outgoing.get(task_id)
    }

    /// Tracks a task delegated to this agent.
    pub fn track_incoming(&mut self, task_id: TaskId, from: AgentId, task_type: String) {
        self.incoming.insert(
            task_id.clone(),
            IncomingTaskInfo {
                task_id,
                from,
                task_type,
                received_at: Instant::now(),
                accepted: false,
            },
        );
    }

    /// Marks an incoming task as accepted.
    pub fn accept_incoming(&mut self, task_id: &TaskId) -> bool {
        if let Some(info) = self.incoming.get_mut(task_id) {
            info.accepted = true;
            true
        } else {
            false
        }
    }

    /// Removes an incoming task (when completed or failed).
    pub fn remove_incoming(&mut self, task_id: &TaskId) -> Option<IncomingTaskInfo> {
        self.incoming.remove(task_id)
    }

    /// Gets information about an incoming task.
    #[must_use]
    pub fn get_incoming(&self, task_id: &TaskId) -> Option<&IncomingTaskInfo> {
        self.incoming.get(task_id)
    }

    /// Returns the number of pending outgoing tasks.
    #[must_use]
    pub fn pending_outgoing_count(&self) -> usize {
        self.outgoing.values().filter(|t| !t.is_terminal()).count()
    }

    /// Returns the number of pending incoming tasks.
    #[must_use]
    pub fn pending_incoming_count(&self) -> usize {
        self.incoming.values().filter(|t| !t.accepted).count()
    }

    /// Removes completed outgoing tasks.
    pub fn cleanup_completed(&mut self) {
        self.outgoing.retain(|_, t| !t.is_terminal());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delegated_task_state_display() {
        assert_eq!(DelegatedTaskState::Pending.to_string(), "pending");
        assert_eq!(DelegatedTaskState::Accepted.to_string(), "accepted");
        assert_eq!(DelegatedTaskState::Completed.to_string(), "completed");
        assert_eq!(DelegatedTaskState::Failed.to_string(), "failed");
    }

    #[test]
    fn delegated_task_creation() {
        let task_id = TaskId::new();
        let agent_id = AgentId::new();
        let task = DelegatedTask::new(task_id.clone(), agent_id.clone(), "code_review".to_string());

        assert_eq!(task.task_id, task_id);
        assert_eq!(task.delegated_to, agent_id);
        assert_eq!(task.task_type, "code_review");
        assert_eq!(task.state, DelegatedTaskState::Pending);
        assert!(task.deadline.is_none());
        assert!(task.result.is_none());
        assert!(task.error.is_none());
    }

    #[test]
    fn delegated_task_with_deadline() {
        let task_id = TaskId::new();
        let agent_id = AgentId::new();
        let task = DelegatedTask::new(task_id, agent_id, "test".to_string())
            .with_deadline(Duration::from_secs(60));

        assert_eq!(task.deadline, Some(Duration::from_secs(60)));
    }

    #[test]
    fn delegated_task_accept() {
        let task_id = TaskId::new();
        let agent_id = AgentId::new();
        let mut task = DelegatedTask::new(task_id, agent_id, "test".to_string());

        task.accept();
        assert_eq!(task.state, DelegatedTaskState::Accepted);
    }

    #[test]
    fn delegated_task_complete() {
        let task_id = TaskId::new();
        let agent_id = AgentId::new();
        let mut task = DelegatedTask::new(task_id, agent_id, "test".to_string());

        let result = serde_json::json!({"success": true});
        task.complete(result.clone());

        assert_eq!(task.state, DelegatedTaskState::Completed);
        assert_eq!(task.result, Some(result));
        assert!(task.is_terminal());
    }

    #[test]
    fn delegated_task_fail() {
        let task_id = TaskId::new();
        let agent_id = AgentId::new();
        let mut task = DelegatedTask::new(task_id, agent_id, "test".to_string());

        task.fail("something went wrong");

        assert_eq!(task.state, DelegatedTaskState::Failed);
        assert_eq!(task.error, Some("something went wrong".to_string()));
        assert!(task.is_terminal());
    }

    #[test]
    fn delegated_task_is_overdue() {
        let task_id = TaskId::new();
        let agent_id = AgentId::new();
        let task = DelegatedTask::new(task_id.clone(), agent_id.clone(), "test".to_string())
            .with_deadline(Duration::from_millis(1));

        // Wait a bit for the deadline to pass
        std::thread::sleep(Duration::from_millis(5));
        assert!(task.is_overdue());

        // Task without deadline is never overdue
        let no_deadline_task = DelegatedTask::new(task_id, agent_id, "test".to_string());
        assert!(!no_deadline_task.is_overdue());
    }

    #[test]
    fn tracker_outgoing_tasks() {
        let mut tracker = DelegationTracker::new();
        let task_id = TaskId::new();
        let agent_id = AgentId::new();

        let task = DelegatedTask::new(task_id.clone(), agent_id.clone(), "code_review".to_string());
        tracker.track_outgoing(task);

        assert!(tracker.get_outgoing(&task_id).is_some());
        assert_eq!(tracker.pending_outgoing_count(), 1);

        // Mark as completed
        let task = tracker.get_outgoing_mut(&task_id).unwrap();
        task.complete(serde_json::json!({}));
        assert_eq!(tracker.pending_outgoing_count(), 0);
    }

    #[test]
    fn tracker_incoming_tasks() {
        let mut tracker = DelegationTracker::new();
        let task_id = TaskId::new();
        let from_agent = AgentId::new();

        tracker.track_incoming(
            task_id.clone(),
            from_agent.clone(),
            "code_review".to_string(),
        );

        assert!(tracker.get_incoming(&task_id).is_some());
        assert_eq!(tracker.pending_incoming_count(), 1);

        // Accept the task
        assert!(tracker.accept_incoming(&task_id));
        assert_eq!(tracker.pending_incoming_count(), 0);

        // Can't accept unknown task
        let unknown_id = TaskId::new();
        assert!(!tracker.accept_incoming(&unknown_id));
    }

    #[test]
    fn tracker_remove_incoming() {
        let mut tracker = DelegationTracker::new();
        let task_id = TaskId::new();
        let from_agent = AgentId::new();

        tracker.track_incoming(task_id.clone(), from_agent, "test".to_string());
        assert!(tracker.get_incoming(&task_id).is_some());

        let removed = tracker.remove_incoming(&task_id);
        assert!(removed.is_some());
        assert!(tracker.get_incoming(&task_id).is_none());
    }

    #[test]
    fn tracker_cleanup_completed() {
        let mut tracker = DelegationTracker::new();
        let task_id1 = TaskId::new();
        let task_id2 = TaskId::new();
        let agent_id = AgentId::new();

        // Add two tasks
        let task1 = DelegatedTask::new(task_id1.clone(), agent_id.clone(), "test1".to_string());
        let task2 = DelegatedTask::new(task_id2.clone(), agent_id, "test2".to_string());
        tracker.track_outgoing(task1);
        tracker.track_outgoing(task2);

        // Complete one task
        tracker
            .get_outgoing_mut(&task_id1)
            .unwrap()
            .complete(serde_json::json!({}));

        // Cleanup removes only completed
        tracker.cleanup_completed();
        assert!(tracker.get_outgoing(&task_id1).is_none());
        assert!(tracker.get_outgoing(&task_id2).is_some());
    }
}

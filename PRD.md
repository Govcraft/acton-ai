
# Rust-Based Agentic AI Framework PRD

## Project Overview

This PRD defines the design for an asynchronous, message-driven AI agentic framework implemented in Rust, leveraging:

* **Acton-Reactive** actor framework for agent interaction.
* **MTI crate** for managing correlation IDs.
* **Tracing crate** for comprehensive instrumentation and observability.

## Objectives

* Enable asynchronous messaging between agents.
* Robust correlation of asynchronous interactions via MTI-generated IDs.
* Centralized context management through a Knowledge Base Agent.
* Advanced instrumentation for observability using Tracing.

## System Architecture

### Agents

* **User Agent**: Task initiator.
* **Agent Manager**: Manages lifecycle, scaling, and errors.
* **Planner Agent**: Decomposes user goals into tasks.
* **Executor Agents**: Perform discrete actions, e.g., LLM tool calls.
* **Knowledge Base Agent**: Stores and retrieves task context and correlation data.
* **Evaluator Agent**: Verifies and evaluates outcomes.
* **Orchestrator Agent**: Manages subscriptions and routes messages via Acton-Reactive.
* **Instrumentation Agent**: Implements tracing for observability.

### Communication Protocol

* Acton-Reactive built-in message broker for asynchronous interactions.
* Messages include:

  * Message type for subscription management.
  * Correlation IDs via MTI for message-response matching.
  * Sender envelope for replies.

### Correlation Handling

* Correlation IDs generated with MTI.
* Knowledge Base Agent stores correlation context.
* Agents retrieve context from Knowledge Base on asynchronous responses.

## Functional Requirements

* **Message Broker**: Acton-Reactive built-in pub/sub system.
* **Agent Management**: Lifecycle and error management.
* **Correlation Management**: MTI-based unique ID generation.
* **Observability**: Comprehensive logging and tracing via Tracing.

## Non-Functional Requirements

* **Scalability**: Horizontal scaling of Executors.
* **Reliability**: Durable messaging and robust error handling.
* **Security**: Secure inter-agent communication and data access controls.

## RACI Matrix

| Task                 | User | Manager | Planner | Executor | Knowledge Base | Evaluator | Orchestrator | Instrumentation |
| -------------------- | ---- | ------- | ------- | -------- | -------------- | --------- | ------------ | --------------- |
| Initiate Tasks       | R/A  | C       | I       | I        | I              | I         | C            | I               |
| Plan Task            | I    | C       | R/A     | C        | C              | I         | C            | I               |
| Execute Task         | I    | C       | C       | R/A      | C              | I         | C            | I               |
| Route Messages       | I    | C       | C       | C        | C              | C         | R/A          | I               |
| Context Management   | I    | C       | C       | C        | R/A            | I         | C            | I               |
| Evaluate Outcomes    | I    | C       | C       | C        | C              | R/A       | C            | I               |
| Logging & Monitoring | I    | C       | I       | I        | I              | I         | C            | R/A             |
| Handle Queries       | I    | R/A     | C       | I        | I              | C         | C            | I               |
| Error Handling       | I/C  | R/A     | C       | C        | C              | C         | C            | R               |
| Replanning & Retries | I/C  | C       | R/A     | C        | C              | C         | C            | I               |
| Broker Control       | I    | C       | I       | I        | I              | I         | R/A          | I               |
| Agent Lifecycle      | C/A  | R       | C       | C        | I              | I         | C            | I               |

## Implementation Considerations

* Use MTI crate for robust correlation IDs.
* Implement Acton-Reactive actor patterns for agent logic and message handling.
* Integrate Tracing crate for structured diagnostics and telemetry.

## Risks and Mitigations

* **Message Loss**: Leverage Acton-Reactive broker reliability features.
* **Correlation ID Collision**: MTI ensures globally unique IDs.
* **Knowledge Base Performance**: Implement distributed or cached storage.

## Success Metrics

* Correlation latency <100ms for 95% of requests.
* Message delivery reliability >99.9%.
* Clear observability with comprehensive tracing logs.

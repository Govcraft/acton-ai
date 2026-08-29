//! Circuit breaking and failover: the state machine, the health query, and
//! the events an operator watches.
//!
//! # Why the state machine is pure
//!
//! Every transition here is a function of `(state, config, outcome, now)`.
//! The [`LLMProvider`](crate::llm::LLMProvider) actor owns one
//! [`BreakerState`] in its model and does nothing to it but replace it with
//! whatever [`next_state`] returns, so the whole of the interesting behavior
//! is testable without spawning anything — the same shape the rate limiter
//! in `provider.rs` already has.
//!
//! # No timers
//!
//! An open circuit carries the instant it may next be probed. Whether it has
//! reached that instant is computed on every observation rather than
//! announced by a timer, so there is no scheduled message to cancel, nothing
//! to leak past the actor's life, and no window in which the recorded state
//! disagrees with the wall clock.
//!
//! # Probes are real traffic
//!
//! A half-open circuit does not send a synthetic request to find out whether
//! the provider recovered: LLM calls cost real money. The next request the
//! caller was going to make anyway is the probe, and its outcome decides.

use acton_reactive::prelude::*;
use std::time::{Duration, Instant};

use crate::llm::config::CircuitBreakerConfig;

/// The recorded state of one provider's circuit breaker.
///
/// Deliberately two variants, not three: "half-open" is not a state anything
/// writes down, it is what [`Open`](Self::Open) *means* once its deadline has
/// passed. Storing it would create a third thing that has to be kept in step
/// with the clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BreakerState {
    /// Requests flow. Carries how many have failed back-to-back so far.
    Closed {
        /// Consecutive failures observed since the last success.
        consecutive_failures: u32,
    },
    /// The circuit is tripped. `until` is when the next request may probe.
    Open {
        /// When this circuit becomes probeable.
        until: Instant,
    },
}

impl Default for BreakerState {
    fn default() -> Self {
        Self::Closed {
            consecutive_failures: 0,
        }
    }
}

/// What a breaker permits right now.
///
/// Derived from a [`BreakerState`] and the current instant by [`phase`];
/// never stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BreakerPhase {
    /// Healthy: dispatch normally.
    Closed,
    /// Tripped: refuse without touching the wire.
    Open,
    /// Cooldown elapsed: the next request is the probe.
    HalfOpen,
}

/// What a breaker permits right now, given the clock.
pub(crate) fn phase(state: BreakerState, now: Instant) -> BreakerPhase {
    match state {
        BreakerState::Closed { .. } => BreakerPhase::Closed,
        BreakerState::Open { until } if now >= until => BreakerPhase::HalfOpen,
        BreakerState::Open { .. } => BreakerPhase::Open,
    }
}

/// The state a breaker moves to after observing one request outcome.
///
/// Pure. The rules, in full:
///
/// - Success always closes the circuit and clears the counter, whether the
///   request was a half-open probe or ordinary traffic.
/// - A failure while closed increments the counter, and trips the circuit
///   once the counter reaches the threshold.
/// - A failure while open — which in practice means a half-open probe
///   failed — re-opens for another full cooldown.
/// - A disabled breaker counts failures but never trips, so switching it on
///   later starts from an honest number rather than from zero.
pub(crate) fn next_state(
    state: BreakerState,
    config: &CircuitBreakerConfig,
    succeeded: bool,
    now: Instant,
) -> BreakerState {
    if succeeded {
        return BreakerState::Closed {
            consecutive_failures: 0,
        };
    }

    match state {
        BreakerState::Closed {
            consecutive_failures,
        } => {
            let failures = consecutive_failures.saturating_add(1);
            if config.enabled && failures >= config.failure_threshold {
                BreakerState::Open {
                    until: now + config.cooldown,
                }
            } else {
                BreakerState::Closed {
                    consecutive_failures: failures,
                }
            }
        }
        BreakerState::Open { until } if !config.enabled => {
            // The breaker was switched off while tripped. Nothing may keep a
            // circuit open that is no longer allowed to trip.
            let _ = until;
            BreakerState::Closed {
                consecutive_failures: config.failure_threshold,
            }
        }
        BreakerState::Open { .. } => BreakerState::Open {
            until: now + config.cooldown,
        },
    }
}

/// Asks a provider actor how its circuit breaker stands.
///
/// Answered from a read-only handler, so it never queues behind the
/// provider's own request processing beyond ordinary mailbox order. The
/// prompt loop asks this only when a failover chain is configured for the
/// provider it resolved — an unconfigured runtime performs no health asks at
/// all.
///
/// ```rust,ignore
/// let health = provider.ask(CheckHealth).await?;
/// ```
#[acton_message]
pub struct CheckHealth;

impl Request for CheckHealth {
    type Response = ProviderHealth;
}

/// A provider's answer to [`CheckHealth`].
#[acton_message]
#[derive(PartialEq, Eq)]
pub enum ProviderHealth {
    /// The circuit is closed: requests are flowing.
    Closed {
        /// Failures observed back-to-back since the last success. Zero on a
        /// provider that has never failed.
        consecutive_failures: u32,
    },
    /// The circuit is tripped and requests are being refused.
    Open {
        /// How much of the cooldown is left before the next probe.
        remaining: Duration,
    },
    /// The cooldown has elapsed: the next request through is the probe.
    HalfOpen,
}

impl ProviderHealth {
    /// Renders a provider's health as a state and, when tripped, how long it
    /// stays that way.
    #[must_use]
    pub(crate) fn from_state(state: BreakerState, now: Instant) -> Self {
        match state {
            BreakerState::Closed {
                consecutive_failures,
            } => Self::Closed {
                consecutive_failures,
            },
            BreakerState::Open { until } if now >= until => Self::HalfOpen,
            BreakerState::Open { until } => Self::Open {
                remaining: until.saturating_duration_since(now),
            },
        }
    }

    /// Whether this provider is refusing requests outright.
    ///
    /// False for a half-open provider: the cooldown has elapsed, so the next
    /// request is a probe rather than a refusal.
    ///
    /// ```rust
    /// use acton_ai::prelude::ProviderHealth;
    /// use std::time::Duration;
    ///
    /// let tripped = ProviderHealth::Open {
    ///     remaining: Duration::from_secs(12),
    /// };
    /// assert!(tripped.is_open());
    /// assert!(tripped.to_string().contains("12.0s"));
    ///
    /// assert!(!ProviderHealth::HalfOpen.is_open());
    /// assert!(!ProviderHealth::Closed {
    ///     consecutive_failures: 3
    /// }
    /// .is_open());
    /// ```
    #[must_use]
    pub fn is_open(&self) -> bool {
        matches!(self, Self::Open { .. })
    }
}

impl std::fmt::Display for ProviderHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed {
                consecutive_failures: 0,
            } => write!(f, "healthy"),
            Self::Closed {
                consecutive_failures,
            } => write!(
                f,
                "healthy after {consecutive_failures} consecutive failure(s)"
            ),
            Self::Open { remaining } => write!(
                f,
                "circuit open for another {:.1}s",
                remaining.as_secs_f64()
            ),
            Self::HalfOpen => write!(f, "circuit half-open; next request probes"),
        }
    }
}

/// Something changed about where requests are going, or which model serves
/// them.
///
/// Broadcast on the broker, so anything that subscribes sees it: the
/// telemetry actor counts them, and
/// [`ActonAIBuilder::on_failover_event`](crate::facade::ActonAIBuilder::on_failover_event)
/// hands them to a caller's closure.
///
/// A deliberately separate enum from
/// [`SystemEvent`](crate::messages::SystemEvent): these four all describe
/// routing decisions an operator is on the hook for, and a subscriber that
/// wants them should not have to filter lifecycle noise out of the same
/// stream.
#[acton_message]
#[derive(PartialEq, Eq)]
pub enum FailoverEvent {
    /// A provider failed enough times in a row to be tripped open. Requests
    /// to it are refused until the cooldown elapses.
    CircuitOpened {
        /// Configured name of the provider whose circuit tripped.
        provider: String,
        /// How many consecutive failures tripped it.
        consecutive_failures: u32,
        /// How long it stays open, in seconds.
        cooldown_secs: u64,
    },
    /// A half-open probe succeeded, so the provider is back in service.
    CircuitClosed {
        /// Configured name of the provider that recovered.
        provider: String,
    },
    /// The prompt loop gave up on one provider for this round and moved to
    /// the next candidate in the chain.
    FailedOver {
        /// The provider that was abandoned for this round.
        from: String,
        /// The provider the round was re-dispatched to.
        to: String,
    },
    /// A rate-limited provider served a request from its fallback model
    /// rather than queueing it.
    ModelDegraded {
        /// Configured name of the provider that degraded.
        provider: String,
        /// The model that would ordinarily have served.
        from_model: String,
        /// The model that actually served.
        to_model: String,
        /// How long the API asked us to wait, in seconds.
        retry_after_secs: u64,
    },
}

impl FailoverEvent {
    /// A short, stable label for this variant.
    ///
    /// A fixed vocabulary rather than a rendered message, because it becomes
    /// a metric attribute — see
    /// `record_failover_event`.
    ///
    /// ```rust
    /// use acton_ai::prelude::FailoverEvent;
    ///
    /// let event = FailoverEvent::FailedOver {
    ///     from: "claude".to_string(),
    ///     to: "local".to_string(),
    /// };
    ///
    /// assert_eq!(event.kind(), "failed_over");
    /// // The provider an alert is about is the one that gave up the round,
    /// // not the one that picked it up.
    /// assert_eq!(event.provider(), "claude");
    /// ```
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::CircuitOpened { .. } => "circuit_opened",
            Self::CircuitClosed { .. } => "circuit_closed",
            Self::FailedOver { .. } => "failed_over",
            Self::ModelDegraded { .. } => "model_degraded",
        }
    }

    /// The provider this event is about.
    ///
    /// For [`FailedOver`](Self::FailedOver) that is the provider being left
    /// behind: it is the one whose failure caused the event.
    #[must_use]
    pub fn provider(&self) -> &str {
        match self {
            Self::CircuitOpened { provider, .. }
            | Self::CircuitClosed { provider }
            | Self::ModelDegraded { provider, .. } => provider,
            Self::FailedOver { from, .. } => from,
        }
    }
}

impl std::fmt::Display for FailoverEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CircuitOpened {
                provider,
                consecutive_failures,
                cooldown_secs,
            } => write!(
                f,
                "circuit opened for provider '{provider}' after {consecutive_failures} \
                 consecutive failures; refusing requests for {cooldown_secs}s"
            ),
            Self::CircuitClosed { provider } => write!(
                f,
                "circuit closed for provider '{provider}'; its probe request succeeded"
            ),
            Self::FailedOver { from, to } => {
                write!(f, "failed over from provider '{from}' to '{to}'")
            }
            Self::ModelDegraded {
                provider,
                from_model,
                to_model,
                retry_after_secs,
            } => write!(
                f,
                "provider '{provider}' is rate limited for {retry_after_secs}s; serving from \
                 '{to_model}' instead of '{from_model}'"
            ),
        }
    }
}

/// A caller-supplied [`FailoverEvent`] handler.
///
/// Wrapped because `#[acton_actor]` derives `Debug` on the model that holds
/// it, and a boxed closure has none of its own.
#[derive(Clone, Default)]
pub(crate) struct FailoverCallback(
    pub(crate) Option<std::sync::Arc<dyn Fn(FailoverEvent) + Send + Sync>>,
);

impl std::fmt::Debug for FailoverCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("FailoverCallback")
            .field(&self.0.is_some())
            .finish()
    }
}

/// Runs a caller's callback for every [`FailoverEvent`] on the broker.
///
/// A whole actor for one closure buys the caller something they cannot get
/// otherwise: broker events are delivered to actors, and the callback has to
/// live somewhere a subscription can reach.
#[acton_actor]
pub(crate) struct FailoverEventListener {
    callback: FailoverCallback,
}

impl FailoverEventListener {
    /// Spawns the listener, subscribed to [`FailoverEvent`] **on the
    /// builder**.
    ///
    /// Subscribing after `start()` is silently ignored, which would leave a
    /// listener that runs happily and never fires.
    pub(crate) async fn spawn(
        runtime: &mut ActorRuntime,
        callback: std::sync::Arc<dyn Fn(FailoverEvent) + Send + Sync>,
    ) -> ActorHandle {
        let mut builder =
            runtime.new_actor_with_name::<FailoverEventListener>("failover_listener".to_string());

        builder.model.callback = FailoverCallback(Some(callback));

        // `act_on`, not `mutate_on`: the callback reads the event and touches
        // no state. It is caller code running on the actor's thread, so it
        // must be cheap and must not block.
        builder.act_on::<FailoverEvent>(|actor, envelope| {
            if let Some(ref callback) = actor.model.callback.0 {
                callback(envelope.message().clone());
            }
            Reply::ready()
        });

        builder.handle().subscribe::<FailoverEvent>().await;

        builder.start().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Threshold 2 so "one more failure" and "the failure that trips it" are
    /// different lines in a test.
    fn breaker() -> CircuitBreakerConfig {
        CircuitBreakerConfig::new(2, Duration::from_secs(30))
    }

    #[test]
    fn a_fresh_breaker_is_closed_with_no_failures() {
        assert_eq!(
            BreakerState::default(),
            BreakerState::Closed {
                consecutive_failures: 0
            }
        );
        assert_eq!(
            phase(BreakerState::default(), Instant::now()),
            BreakerPhase::Closed
        );
    }

    #[test]
    fn failures_below_the_threshold_only_count() {
        let now = Instant::now();

        let state = next_state(BreakerState::default(), &breaker(), false, now);

        assert_eq!(
            state,
            BreakerState::Closed {
                consecutive_failures: 1
            }
        );
        assert_eq!(phase(state, now), BreakerPhase::Closed);
    }

    #[test]
    fn reaching_the_threshold_opens_the_circuit() {
        let now = Instant::now();
        let config = breaker();

        let state = next_state(
            BreakerState::Closed {
                consecutive_failures: 1,
            },
            &config,
            false,
            now,
        );

        assert_eq!(
            state,
            BreakerState::Open {
                until: now + config.cooldown
            }
        );
        assert_eq!(phase(state, now), BreakerPhase::Open);
    }

    #[test]
    fn a_success_resets_the_counter() {
        let now = Instant::now();

        let state = next_state(
            BreakerState::Closed {
                consecutive_failures: 1,
            },
            &breaker(),
            true,
            now,
        );

        assert_eq!(
            state,
            BreakerState::Closed {
                consecutive_failures: 0
            },
            "a success between failures must not leave the next one one-from-tripping"
        );
    }

    #[test]
    fn an_open_circuit_becomes_half_open_at_its_deadline() {
        let now = Instant::now();
        let state = BreakerState::Open {
            until: now + Duration::from_secs(30),
        };

        assert_eq!(phase(state, now), BreakerPhase::Open);
        assert_eq!(
            phase(state, now + Duration::from_secs(29)),
            BreakerPhase::Open
        );
        assert_eq!(
            phase(state, now + Duration::from_secs(30)),
            BreakerPhase::HalfOpen,
            "the deadline itself must already admit the probe"
        );
    }

    #[test]
    fn a_successful_probe_closes_the_circuit() {
        let now = Instant::now();
        let open = BreakerState::Open { until: now };

        let state = next_state(open, &breaker(), true, now);

        assert_eq!(
            state,
            BreakerState::Closed {
                consecutive_failures: 0
            }
        );
    }

    #[test]
    fn a_failed_probe_opens_the_circuit_for_another_full_cooldown() {
        let now = Instant::now();
        let config = breaker();
        let open = BreakerState::Open { until: now };

        let state = next_state(open, &config, false, now);

        assert_eq!(
            state,
            BreakerState::Open {
                until: now + config.cooldown
            },
            "a failed probe must not leave a deadline that is already in the past"
        );
        assert_eq!(phase(state, now), BreakerPhase::Open);
    }

    #[test]
    fn a_disabled_breaker_never_opens() {
        let now = Instant::now();
        let config = CircuitBreakerConfig::disabled();

        let mut state = BreakerState::default();
        for _ in 0..100 {
            state = next_state(state, &config, false, now);
            assert_eq!(phase(state, now), BreakerPhase::Closed);
        }

        assert_eq!(
            state,
            BreakerState::Closed {
                consecutive_failures: 100
            },
            "failures are still counted, so switching the breaker on later starts from the truth"
        );
    }

    #[test]
    fn disabling_a_breaker_releases_a_circuit_it_had_already_tripped() {
        let now = Instant::now();
        let open = BreakerState::Open {
            until: now + Duration::from_secs(3600),
        };

        let state = next_state(open, &CircuitBreakerConfig::disabled(), false, now);

        assert_eq!(phase(state, now), BreakerPhase::Closed);
    }

    #[test]
    fn a_zero_cooldown_is_probeable_immediately() {
        let now = Instant::now();
        let config = CircuitBreakerConfig::new(1, Duration::ZERO);

        let state = next_state(BreakerState::default(), &config, false, now);

        assert_eq!(state, BreakerState::Open { until: now });
        assert_eq!(
            phase(state, now),
            BreakerPhase::HalfOpen,
            "a zero cooldown means the very next request probes"
        );
    }

    #[test]
    fn health_reports_the_state_the_breaker_is_in() {
        let now = Instant::now();

        assert_eq!(
            ProviderHealth::from_state(BreakerState::default(), now),
            ProviderHealth::Closed {
                consecutive_failures: 0
            }
        );
        assert_eq!(
            ProviderHealth::from_state(
                BreakerState::Open {
                    until: now + Duration::from_secs(10)
                },
                now
            ),
            ProviderHealth::Open {
                remaining: Duration::from_secs(10)
            }
        );
        assert_eq!(
            ProviderHealth::from_state(BreakerState::Open { until: now }, now),
            ProviderHealth::HalfOpen
        );
    }

    #[test]
    fn only_an_open_circuit_reports_itself_open() {
        let now = Instant::now();

        assert!(ProviderHealth::from_state(
            BreakerState::Open {
                until: now + Duration::from_secs(1)
            },
            now
        )
        .is_open());
        assert!(
            !ProviderHealth::from_state(BreakerState::Open { until: now }, now).is_open(),
            "half-open admits the probe, so it must not read as refusing"
        );
        assert!(!ProviderHealth::from_state(BreakerState::default(), now).is_open());
    }

    #[test]
    fn health_renders_what_it_means() {
        assert_eq!(
            ProviderHealth::Closed {
                consecutive_failures: 0
            }
            .to_string(),
            "healthy"
        );
        assert!(ProviderHealth::Closed {
            consecutive_failures: 3
        }
        .to_string()
        .contains('3'));
        assert!(ProviderHealth::Open {
            remaining: Duration::from_secs(12)
        }
        .to_string()
        .contains("12"));
        assert!(ProviderHealth::HalfOpen.to_string().contains("half-open"));
    }

    #[test]
    fn every_event_reports_a_kind_and_the_provider_it_is_about() {
        let opened = FailoverEvent::CircuitOpened {
            provider: "claude".to_string(),
            consecutive_failures: 5,
            cooldown_secs: 30,
        };
        assert_eq!(opened.kind(), "circuit_opened");
        assert_eq!(opened.provider(), "claude");

        let closed = FailoverEvent::CircuitClosed {
            provider: "claude".to_string(),
        };
        assert_eq!(closed.kind(), "circuit_closed");
        assert_eq!(closed.provider(), "claude");

        let failed_over = FailoverEvent::FailedOver {
            from: "claude".to_string(),
            to: "backup".to_string(),
        };
        assert_eq!(failed_over.kind(), "failed_over");
        assert_eq!(
            failed_over.provider(),
            "claude",
            "the event is about the provider that failed, not the one that rescued it"
        );

        let degraded = FailoverEvent::ModelDegraded {
            provider: "claude".to_string(),
            from_model: "opus".to_string(),
            to_model: "haiku".to_string(),
            retry_after_secs: 7,
        };
        assert_eq!(degraded.kind(), "model_degraded");
        assert_eq!(degraded.provider(), "claude");
    }

    #[test]
    fn events_render_the_whole_story() {
        let opened = FailoverEvent::CircuitOpened {
            provider: "claude".to_string(),
            consecutive_failures: 5,
            cooldown_secs: 30,
        }
        .to_string();
        assert!(opened.contains("claude"), "{opened}");
        assert!(opened.contains('5'), "{opened}");
        assert!(opened.contains("30s"), "{opened}");

        let degraded = FailoverEvent::ModelDegraded {
            provider: "claude".to_string(),
            from_model: "opus".to_string(),
            to_model: "haiku".to_string(),
            retry_after_secs: 7,
        }
        .to_string();
        assert!(degraded.contains("opus"), "{degraded}");
        assert!(
            degraded.contains("haiku"),
            "nobody should learn from a bill that opus became haiku: {degraded}"
        );

        let failed_over = FailoverEvent::FailedOver {
            from: "claude".to_string(),
            to: "backup".to_string(),
        }
        .to_string();
        assert!(failed_over.contains("claude"), "{failed_over}");
        assert!(failed_over.contains("backup"), "{failed_over}");

        assert!(FailoverEvent::CircuitClosed {
            provider: "claude".to_string()
        }
        .to_string()
        .contains("claude"));
    }
}

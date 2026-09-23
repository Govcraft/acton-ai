//! [`EventTime`] and [`TimeSource`]: what the runtime means by "now".

use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use chrono::{DateTime, SecondsFormat, SubsecRound, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::clock::ClockError;

/// A point on the runtime's timeline, in UTC, to the microsecond.
///
/// Truncated to microseconds on construction so that the RFC 3339 rendering
/// written to a journal parses back to exactly the same value. A replay that
/// compared a nanosecond original with its microsecond echo would find every
/// timestamp "different", which is the one outcome a replay must never have.
///
/// Serialized as an RFC 3339 string with a `Z` suffix, the same shape the
/// audit trail already writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventTime(DateTime<Utc>);

impl EventTime {
    /// Wraps a UTC instant, truncated to the microsecond.
    #[must_use]
    pub fn from_utc(at: DateTime<Utc>) -> Self {
        Self(at.trunc_subsecs(6))
    }

    /// The wall clock right now.
    ///
    /// Only [`TimeSource::system`] should call this. Everything else asks a
    /// [`TimeSource`], which is what lets a replay or a test decide the answer.
    #[must_use]
    pub(crate) fn wall_clock() -> Self {
        Self::from_utc(Utc::now())
    }

    /// Parses an RFC 3339 timestamp.
    ///
    /// # Errors
    ///
    /// Returns [`ClockErrorKind::InvalidTime`](crate::clock::ClockErrorKind::InvalidTime)
    /// naming the text when it is not RFC 3339.
    pub fn parse_rfc3339(text: &str) -> Result<Self, ClockError> {
        DateTime::parse_from_rfc3339(text)
            .map(|at| Self::from_utc(at.with_timezone(&Utc)))
            .map_err(|error| ClockError::invalid_time(text, error.to_string()))
    }

    /// The instant as a [`DateTime<Utc>`].
    #[must_use]
    pub fn as_utc(&self) -> DateTime<Utc> {
        self.0
    }

    /// RFC 3339 with microseconds and a `Z` suffix.
    #[must_use]
    pub fn to_rfc3339(&self) -> String {
        self.0.to_rfc3339_opts(SecondsFormat::Micros, true)
    }

    /// The `YYYY-MM-DD HH:MM:SS` text SQLite's `datetime('now')` produces.
    ///
    /// Stored timestamps are compared as text (`next_due <= ?`), so a value
    /// bound in place of `datetime('now')` has to be rendered in exactly that
    /// format or it would sort against existing rows by accident rather than
    /// by time.
    #[must_use]
    pub fn to_sql(&self) -> String {
        self.0.format("%Y-%m-%d %H:%M:%S").to_string()
    }
}

impl fmt::Display for EventTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_rfc3339())
    }
}

impl Serialize for EventTime {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_rfc3339())
    }
}

impl<'de> Deserialize<'de> for EventTime {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Self::parse_rfc3339(&text).map_err(serde::de::Error::custom)
    }
}

/// Where the runtime reads the current time.
///
/// Cheap to clone; every clone reads the same timeline.
///
/// - [`TimeSource::system`] is the wall clock, and the default. A live
///   process uses it.
/// - [`TimeSource::driven`] stands still until it is moved with
///   [`advance_to`](Self::advance_to). A replay moves it to each recorded
///   tick in turn, so every timestamp the runtime writes while handling that
///   tick is the recorded time rather than today's. A test moves it to run a
///   week of schedule in a few milliseconds.
///
/// # Not the scheduling clock
///
/// acton-reactive has its own `Clock`, which measures *durations* for
/// scheduled sends against a monotonic [`Instant`](std::time::Instant). This
/// is the other kind of time: the calendar instant that lands in a record,
/// is compared against a `next_due`, or is replayed from a journal. Timeouts,
/// backoff, and circuit-breaker cooldowns stay on `Instant` and are not
/// affected by anything here.
#[derive(Clone, Default)]
pub struct TimeSource {
    inner: Source,
}

#[derive(Clone, Default)]
enum Source {
    #[default]
    System,
    Driven(Arc<Mutex<EventTime>>),
}

impl TimeSource {
    /// The wall clock.
    #[must_use]
    pub fn system() -> Self {
        Self {
            inner: Source::System,
        }
    }

    /// A timeline that starts at `start` and moves only when told to.
    #[must_use]
    pub fn driven(start: EventTime) -> Self {
        Self {
            inner: Source::Driven(Arc::new(Mutex::new(start))),
        }
    }

    /// The current time on this timeline.
    #[must_use]
    pub fn now(&self) -> EventTime {
        match &self.inner {
            Source::System => EventTime::wall_clock(),
            // A poisoned lock still holds a valid timestamp: the only write
            // is a single assignment, which cannot be left half-done.
            Source::Driven(at) => *at.lock().unwrap_or_else(PoisonError::into_inner),
        }
    }

    /// Whether this timeline only moves when [`advance_to`](Self::advance_to)
    /// moves it.
    #[must_use]
    pub fn is_driven(&self) -> bool {
        matches!(self.inner, Source::Driven(_))
    }

    /// Moves a driven timeline to `at`.
    ///
    /// Staying put is allowed — two ticks may share a timestamp — but going
    /// backwards is not: a journal whose times run backwards would produce
    /// `next_due` comparisons that differ from the run that wrote it.
    ///
    /// # Errors
    ///
    /// Returns [`ClockErrorKind::NotDriven`](crate::clock::ClockErrorKind::NotDriven)
    /// on the wall clock, which nobody can move, and
    /// [`ClockErrorKind::TimeWentBackwards`](crate::clock::ClockErrorKind::TimeWentBackwards)
    /// when `at` is earlier than the current time.
    pub fn advance_to(&self, at: EventTime) -> Result<(), ClockError> {
        let Source::Driven(current) = &self.inner else {
            return Err(ClockError::not_driven());
        };
        let mut current = current.lock().unwrap_or_else(PoisonError::into_inner);
        if at < *current {
            return Err(ClockError::time_went_backwards(*current, at));
        }
        *current = at;
        Ok(())
    }
}

impl fmt::Debug for TimeSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            Source::System => f.write_str("TimeSource::System"),
            Source::Driven(_) => f
                .debug_tuple("TimeSource::Driven")
                .field(&self.now())
                .finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ClockErrorKind;

    fn at(text: &str) -> EventTime {
        EventTime::parse_rfc3339(text).expect("valid timestamp")
    }

    #[test]
    fn rfc3339_round_trips_exactly() {
        // Nanoseconds in, microseconds kept: the rendering must parse back to
        // an equal value, or a replay would see every time as changed.
        let original = EventTime::from_utc(
            DateTime::parse_from_rfc3339("2026-09-23T12:34:56.123456789Z")
                .unwrap()
                .with_timezone(&Utc),
        );
        assert_eq!(original.to_rfc3339(), "2026-09-23T12:34:56.123456Z");
        assert_eq!(at(&original.to_rfc3339()), original);
    }

    #[test]
    fn offsets_are_normalised_to_utc() {
        assert_eq!(at("2026-09-23T14:00:00+02:00"), at("2026-09-23T12:00:00Z"),);
    }

    #[test]
    fn sql_rendering_matches_sqlite_datetime() {
        assert_eq!(at("2026-09-23T07:08:09.5Z").to_sql(), "2026-09-23 07:08:09");
    }

    #[test]
    fn serde_uses_rfc3339_text() {
        let time = at("2026-09-23T12:00:00Z");
        let json = serde_json::to_string(&time).unwrap();
        assert_eq!(json, "\"2026-09-23T12:00:00.000000Z\"");
        assert_eq!(serde_json::from_str::<EventTime>(&json).unwrap(), time);
    }

    #[test]
    fn unparseable_text_names_itself() {
        let error = EventTime::parse_rfc3339("yesterday").unwrap_err();
        assert!(matches!(error.kind(), ClockErrorKind::InvalidTime { .. }));
        assert!(error.to_string().contains("yesterday"), "{error}");
    }

    #[test]
    fn a_driven_source_stands_still_until_advanced() {
        let source = TimeSource::driven(at("2026-01-01T00:00:00Z"));
        assert!(source.is_driven());
        assert_eq!(source.now(), at("2026-01-01T00:00:00Z"));

        source.advance_to(at("2026-01-08T00:00:00Z")).unwrap();
        assert_eq!(source.now(), at("2026-01-08T00:00:00Z"));
    }

    #[test]
    fn clones_share_one_timeline() {
        let source = TimeSource::driven(at("2026-01-01T00:00:00Z"));
        let clone = source.clone();
        source.advance_to(at("2026-01-02T00:00:00Z")).unwrap();
        assert_eq!(clone.now(), at("2026-01-02T00:00:00Z"));
    }

    #[test]
    fn a_driven_source_refuses_to_go_backwards_but_may_stay_put() {
        let source = TimeSource::driven(at("2026-01-02T00:00:00Z"));
        source.advance_to(at("2026-01-02T00:00:00Z")).unwrap();

        let error = source.advance_to(at("2026-01-01T00:00:00Z")).unwrap_err();
        assert!(matches!(
            error.kind(),
            ClockErrorKind::TimeWentBackwards { .. }
        ));
        assert_eq!(source.now(), at("2026-01-02T00:00:00Z"), "unchanged");
    }

    #[test]
    fn the_system_source_cannot_be_moved() {
        let source = TimeSource::system();
        assert!(!source.is_driven());
        let error = source.advance_to(at("2026-01-01T00:00:00Z")).unwrap_err();
        assert_eq!(error.kind(), &ClockErrorKind::NotDriven);
    }

    #[test]
    fn the_default_is_the_wall_clock() {
        assert!(!TimeSource::default().is_driven());
    }
}

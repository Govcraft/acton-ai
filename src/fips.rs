//! Process-wide TLS crypto provider selection.
//!
//! rustls does not pick a cryptography backend for you when more than one
//! could be linked; it asks for a process-wide default. This module installs
//! that default, and under the `fips` feature it installs the FIPS 140-3
//! validated AWS-LC module rather than the ordinary one.
//!
//! # Why it has to be first
//!
//! The provider is a process-wide singleton, and rustls only honours the
//! first installation. Anything that opens a TLS connection before
//! [`install_crypto_provider`] runs has already picked a backend, and no later
//! call can take it back. So it runs at the top of `main`, before the CLI
//! parses its own arguments, and again — idempotently — from
//! [`ActonAIBuilder::launch`](crate::ActonAIBuilder::launch) for embedders who
//! never go through this crate's `main`.
//!
//! # Without the feature
//!
//! [`install_crypto_provider`] is a no-op that returns `Ok(())`. reqwest's
//! ring-backed stack installs its own provider on first use, exactly as it did
//! before this module existed.
//!
//! # Building
//!
//! ```text
//! cargo build --release --no-default-features \
//!     --features "fips,sandbox-hardening,derive,otel,ipc"
//! ```
//!
//! Release, not debug: AWS-LC's power-on integrity test hashes its own loaded
//! text segment, and a debug build's relocations change that hash between
//! runs, so the module aborts the process at startup. That is the FIPS module
//! working as specified, not a bug in this crate.

use crate::error::ActonAIError;

/// Installs the process-wide rustls crypto provider.
///
/// Idempotent: calling it twice is fine, and calling it after something else
/// already installed the same kind of provider is fine. What is *not* fine —
/// and what this reports as an error under the `fips` feature — is finding a
/// non-FIPS provider already installed, because a build that exists to
/// guarantee FIPS-validated cryptography must not quietly run on something
/// else.
///
/// # Errors
///
/// Returns a configuration error if a non-FIPS provider was already installed
/// in this process, or if the FIPS provider reports that it is not operating
/// in FIPS mode.
#[cfg(feature = "fips")]
pub fn install_crypto_provider() -> Result<(), ActonAIError> {
    let provider = rustls::crypto::default_fips_provider();

    // Asked, not assumed. The provider type is the same either way; only this
    // flag distinguishes a module running its validated algorithms from one
    // that fell back.
    if !provider.fips() {
        return Err(ActonAIError::configuration(
            "fips",
            "the aws-lc-rs provider reports that it is not running in FIPS mode; this build \
             claims FIPS-validated cryptography and must not start without it",
        ));
    }

    match provider.install_default() {
        Ok(()) => {
            tracing::info!("installed the FIPS crypto provider as the process-wide rustls default");
            Ok(())
        }
        // Already installed. Ours or somebody else's — the only question that
        // matters is whether what is installed is FIPS.
        Err(existing) => {
            if existing.fips() {
                Ok(())
            } else {
                Err(ActonAIError::configuration(
                    "fips",
                    "a non-FIPS rustls crypto provider was already installed in this process; \
                     call acton_ai::fips::install_crypto_provider() before anything opens a TLS \
                     connection",
                ))
            }
        }
    }
}

/// Installs the process-wide rustls crypto provider.
///
/// A no-op without the `fips` feature: reqwest installs the ring-backed
/// provider itself on first use, and there is nothing here to choose between.
///
/// # Errors
///
/// Never, in this build. The signature matches the `fips` build so callers
/// need no `cfg` of their own.
#[cfg(not(feature = "fips"))]
pub fn install_crypto_provider() -> Result<(), ActonAIError> {
    Ok(())
}

/// Whether this build routes TLS through the FIPS-validated provider.
///
/// Reported rather than inferred from a feature flag at the call site, so a
/// status line or a support bundle can say which binary it is looking at.
#[must_use]
pub const fn is_fips_build() -> bool {
    cfg!(feature = "fips")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn installing_twice_is_not_an_error() {
        install_crypto_provider().expect("the first install must succeed");
        install_crypto_provider().expect("a second install must be a no-op, not a failure");
    }

    #[test]
    fn the_build_reports_its_own_shape() {
        assert_eq!(is_fips_build(), cfg!(feature = "fips"));
    }
}

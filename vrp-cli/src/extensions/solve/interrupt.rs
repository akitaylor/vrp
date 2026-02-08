//! Ctrl-C interruption helpers for solver execution.

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use vrp_core::utils::{Float, InfoLogger, Quota, TimeQuota};

static SHOULD_INTERRUPT: AtomicBool = AtomicBool::new(false);
static INTERRUPT_LOGGER: OnceLock<InfoLogger> = OnceLock::new();

/// Creates interruption quota which reacts to max time and Ctrl-C.
pub fn create_interruption_quota(max_time: Option<usize>, logger: InfoLogger) -> Arc<dyn Quota> {
    struct InterruptionQuota {
        inner: Option<Arc<dyn Quota>>,
    }

    impl Quota for InterruptionQuota {
        fn is_reached(&self) -> bool {
            self.inner.as_ref().is_some_and(|inner| inner.is_reached()) || SHOULD_INTERRUPT.load(Ordering::Relaxed)
        }
    }

    let inner = max_time.map::<Arc<dyn Quota>, _>(|time| Arc::new(TimeQuota::new(time as Float)));
    let _ = INTERRUPT_LOGGER.set(logger);

    // NOTE ignore error which happens when handler already installed.
    let _ = ctrlc::set_handler({
        move || {
            let was_set = SHOULD_INTERRUPT.swap(true, Ordering::Relaxed);
            if !was_set {
                if let Some(logger) = INTERRUPT_LOGGER.get() {
                    (logger)("interrupt received, stopping after current generation...");
                } else {
                    eprintln!("interrupt received, stopping after current generation...");
                }
            }
        }
    });

    Arc::new(InterruptionQuota { inner })
}

import time
import logging

logger = logging.getLogger(__name__)

def retry_with_backoff(
    func,
    max_attempts=3,
    initial_delay=1,
    backoff_factor=2
):
    for attempt in range(1, max_attempts + 1):
        try:
            return func()

        except Exception as error:
            if attempt == max_attempts:
                logger.error(
                    "All retry attempts failed: %s",
                    error,
                )
                raise

            delay = initial_delay * (backoff_factor ** (attempt -1))

            logger.warning(
                "Attemp %d failed: %s. Retrying in %d seconds...",
                attempt,
                error,
                delay,
            )

            time.sleep(delay)

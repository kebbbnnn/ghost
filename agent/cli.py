"""CLI entry point for the Screen Observation Agent.

Usage:
    python3 -m agent.cli \\
        --app "Cursor" \\
        --instruction "When an error modal appears, click the Retry button" \\
        --instruction "When a cookie consent banner appears, click Accept"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from agent.orchestrator import ObservationAgent


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the agent."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noise from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="screen-agent",
        description="Screen Observation Agent — watches a target app and acts on instructions.",
    )
    parser.add_argument(
        "--app",
        required=True,
        help="Name of the target application to observe (e.g., 'Cursor', 'Google Chrome').",
    )
    parser.add_argument(
        "--instruction", "-i",
        action="append",
        required=True,
        dest="instructions",
        help="Condition → action rule. Can be specified multiple times.",
    )
    parser.add_argument(
        "--proxy-url",
        default="http://127.0.0.1:8100/v1/chat/completions",
        help="Ghost proxy URL (default: http://127.0.0.1:8100/v1/chat/completions).",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help="Model to use for LLM calls (default: auto).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    """Run the agent loop with graceful shutdown."""
    agent = ObservationAgent(
        app_name=args.app,
        instructions=args.instructions,
        proxy_url=args.proxy_url,
        model=args.model,
    )

    # Set up graceful shutdown on Ctrl+C
    loop = asyncio.get_running_loop()

    def _shutdown():
        print("\n⏹️  Stopping agent...")
        agent.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    await agent.run()

    # Print final summary
    print(f"\n📊 Final stats: {agent.stats.summary()}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Screen Observation Agent")
    logger.info("=" * 60)
    logger.info("Target app: %s", args.app)
    logger.info("Instructions: %d", len(args.instructions))
    logger.info("Proxy: %s", args.proxy_url)
    logger.info("Model: %s", args.model)
    logger.info("=" * 60)

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

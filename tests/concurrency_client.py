from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import httpx


DEFAULT_QUESTIONS = [
    "Transformer 的主要機制是什麼？",
    "Transformer 如何處理序列中的 token 關係？",
    "Self-attention 的用途是什麼？",
    "Multi-head attention 有什麼作用？",
]


@dataclass(frozen=True, slots=True)
class RequestResult:
    user_id: int
    status_code: int | None
    elapsed_ms: float
    answer: str | None
    server_total_ms: float | None
    error: str | None


async def query_user(
    *,
    user_id: int,
    client: httpx.AsyncClient,
    endpoint: str,
    question: str,
    top_k: int,
    start_event: asyncio.Event,
) -> RequestResult:
    # 讓所有 user 都建立完成後再一起送出。
    await start_event.wait()

    started_at = perf_counter()

    try:
        response = await client.post(
            endpoint,
            json={
                "question": question,
                "top_k": top_k,
            },
        )

        elapsed_ms = (
            perf_counter() - started_at
        ) * 1000.0

        response.raise_for_status()

        body: dict[str, Any] = response.json()

        timings = body.get("timings", {})
        server_total_ms = timings.get("total_ms")

        return RequestResult(
            user_id=user_id,
            status_code=response.status_code,
            elapsed_ms=elapsed_ms,
            answer=body.get("answer"),
            server_total_ms=(
                float(server_total_ms)
                if server_total_ms is not None
                else None
            ),
            error=None,
        )

    except Exception as exc:
        elapsed_ms = (
            perf_counter() - started_at
        ) * 1000.0

        status_code = None

        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code

        return RequestResult(
            user_id=user_id,
            status_code=status_code,
            elapsed_ms=elapsed_ms,
            answer=None,
            server_total_ms=None,
            error=str(exc),
        )


async def run_test(
    *,
    base_url: str,
    users: int,
    top_k: int,
    timeout_sec: float,
    question: str | None,
) -> int:
    if not 1 <= users <= 4:
        raise ValueError(
            "users must be between 1 and 4"
        )

    endpoint = f"{base_url.rstrip('/')}/query"
    start_event = asyncio.Event()

    timeout = httpx.Timeout(
        timeout_sec,
        connect=10.0,
    )

    limits = httpx.Limits(
        max_connections=4,
        max_keepalive_connections=4,
    )

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
    ) as client:
        tasks = []

        for index in range(users):
            selected_question = (
                question
                if question is not None
                else DEFAULT_QUESTIONS[
                    index % len(DEFAULT_QUESTIONS)
                ]
            )

            tasks.append(
                asyncio.create_task(
                    query_user(
                        user_id=index + 1,
                        client=client,
                        endpoint=endpoint,
                        question=selected_question,
                        top_k=top_k,
                        start_event=start_event,
                    )
                )
            )

        print(
            f"Starting {users} concurrent user(s)..."
        )

        total_started_at = perf_counter()

        # 釋放所有等待中的 user。
        start_event.set()

        results = await asyncio.gather(*tasks)

        total_elapsed_ms = (
            perf_counter() - total_started_at
        ) * 1000.0

    success_count = 0

    print()
    print("=== Results ===")

    for result in results:
        print()
        print(f"User {result.user_id}")
        print(f"  status: {result.status_code}")
        print(
            f"  client elapsed: "
            f"{result.elapsed_ms:.2f} ms"
        )

        if result.server_total_ms is not None:
            print(
                f"  server total: "
                f"{result.server_total_ms:.2f} ms"
            )

        if result.error is not None:
            print(f"  error: {result.error}")
            continue

        success_count += 1
        print(f"  answer: {result.answer}")

    print()
    print("=== Summary ===")
    print(f"users: {users}")
    print(f"success: {success_count}")
    print(f"failed: {users - success_count}")
    print(
        f"wall-clock total: {total_elapsed_ms:.2f} ms"
    )

    successful_results = [
        result
        for result in results
        if result.error is None
    ]

    if successful_results:
        average_ms = sum(
            result.elapsed_ms
            for result in successful_results
        ) / len(successful_results)

        maximum_ms = max(
            result.elapsed_ms
            for result in successful_results
        )

        print(
            f"average client latency: "
            f"{average_ms:.2f} ms"
        )
        print(
            f"maximum client latency: "
            f"{maximum_ms:.2f} ms"
        )

    return 0 if success_count == users else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send up to four concurrent requests "
            "to the RAG service."
        )
    )

    parser.add_argument(
        "--users",
        type=int,
        default=2,
        help="Number of concurrent users, 1 to 4.",
    )

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
        help="RAG service base URL.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds.",
    )

    parser.add_argument(
        "--question",
        default=None,
        help=(
            "Use the same question for every user. "
            "If omitted, each user receives a "
            "different predefined question."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 1 <= args.users <= 4:
        raise SystemExit(
            "--users must be between 1 and 4"
        )

    if args.top_k <= 0:
        raise SystemExit(
            "--top-k must be greater than 0"
        )

    exit_code = asyncio.run(
        run_test(
            base_url=args.base_url,
            users=args.users,
            top_k=args.top_k,
            timeout_sec=args.timeout,
            question=args.question,
        )
    )

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
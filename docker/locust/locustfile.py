import os
import time
import json
import logging
import requests
from locust import User, task, events, between

logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("MODEL_ID", "zaid/gemma-2-9b-awq")
VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000")
STREAM_TIMEOUT = float(os.getenv("STREAM_TIMEOUT", "120"))


class LLMLoadUser(User):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        self.session = requests.Session()

    @task
    def stream_completion(self):
        payload = {
            "model": MODEL_ID,
            "prompt": "Write a short paragraph about attention mechanisms in transformers.",
            "max_tokens": 200,
            "temperature": 0.7,
            "stream": True,
        }

        ttft = 0.0
        itl_values = []
        total_tokens = 0
        request_start = time.monotonic()
        first_token_time = None
        prev_token_time = None

        try:
            response = self.session.post(
                f"{VLLM_URL}/v1/completions",
                json=payload,
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=STREAM_TIMEOUT,
            )

            if response.status_code != 200:
                logger.error(f"Request failed: {response.status_code}")
                events.request.fire(
                    request_type="POST",
                    name="/v1/completions (stream)",
                    response_time=int((time.monotonic() - request_start) * 1000),
                    response_length=0,
                    exception=Exception(f"HTTP {response.status_code}"),
                )
                return

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break

                now = time.monotonic()

                if first_token_time is None:
                    first_token_time = now
                    ttft = now - request_start
                    prev_token_time = now
                else:
                    itl = now - prev_token_time
                    itl_values.append(itl)
                    prev_token_time = now

                total_tokens += 1

            e2e_latency = time.monotonic() - request_start
            response_time_ms = int(e2e_latency * 1000)

            events.request.fire(
                request_type="POST",
                name="/v1/completions (stream)",
                response_time=response_time_ms,
                response_length=total_tokens,
            )

            if first_token_time is not None:
                ttft_ms = int(ttft * 1000)
                events.request.fire(
                    request_type="CUSTOM",
                    name="TTFT",
                    response_time=ttft_ms,
                    response_length=1,
                )

            avg_itl = sum(itl_values) / len(itl_values) if itl_values else 0
            if avg_itl > 0:
                itl_ms = int(avg_itl * 1000)
                events.request.fire(
                    request_type="CUSTOM",
                    name="ITL (avg)",
                    response_time=itl_ms,
                    response_length=1,
                )

        except requests.exceptions.Timeout:
            logger.error("Timeout during streaming")
        except Exception as e:
            logger.error(f"Error: {e}")

    def on_stop(self):
        self.session.close()

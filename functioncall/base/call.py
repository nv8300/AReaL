import asyncio
import logging
import os
import random
import time
import traceback
from statistics import median
from typing import Any, Dict

import aiohttp

from functioncall.base import logging

logger = logging.getLogger("Functioncall")


FUNCTIONCALL_SERVICE_DOMAIN = os.getenv(
    "FUNCTIONCALL_SERVICE_DOMAIN",
    "",
)


def calculate_percentile(elapsed_times, percentile):
    sorted_times = sorted(elapsed_times)
    index = int(len(sorted_times) * (percentile / 100))
    return sorted_times[min(index, len(sorted_times) - 1)]


async def async_invoke_function(
    session: aiohttp.ClientSession,
    function_name: str,
    timeout: aiohttp.ClientTimeout,
    payload: Dict[str, Any] = None,
    max_retries: int = 3,
    initial_retry_interval: float = 0.1,
    max_retry_interval: float = 10.0,
):
    if payload is None:
        payload = {}
    url = f"{FUNCTIONCALL_SERVICE_DOMAIN}/hapis/faas.hcs.io/v1/functions/{function_name}/invoke"
    params = {"invocationType": "RequestResponse"}

    retries = 0
    while retries < max_retries:
        try:
            async with session.post(
                url,
                params=params,
                json=payload,
                timeout=timeout,
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(
                        f"HTTP Error {response.status}: {text} : {response.headers}"
                    )

                try:
                    result = await response.json()
                    return result, response.headers
                except aiohttp.ContentTypeError as e:
                    raise Exception("Invalid JSON response") from e

        except asyncio.TimeoutError as e:
            logger.warning(
                f"Request timeout after {timeout}s, URL: {url}, Headers: {session.headers}, payload: {payload}"
            )
            break

        except Exception as e:
            logger.error(f"Async invocation failed on attempt {retries + 1}:{str(e)}, URL: {url}, Headers: {session.headers}")

        retries += 1
        if retries > max_retries:
            return None, None

        # 指数退避 + 随机抖动
        sleep_time = min(
            initial_retry_interval * (2**retries) + random.uniform(0, 0.1),
            max_retry_interval,
        )
        await asyncio.sleep(sleep_time)


async def batch_function_call_async(
    payload_list, function_name, timeout, concurrency=1000
):
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        semaphore = asyncio.Semaphore(concurrency)

        async def limited_task(payload):
            if not payload:
                return None
            async with semaphore:
                st = time.monotonic()
                result = await async_invoke_function(
                    session, function_name, timeout, payload
                )
                return result, time.monotonic() - st

        tasks = [limited_task(payload) for payload in payload_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        data_list = []
        elapsed_times = []
        max_elapsed = -1
        max_elapsed_header = None
        for (data, header), elapsed in results:
            if elapsed > max_elapsed:
                max_elapsed = elapsed
                max_elapsed_header = header
            data_list.append(data)
            elapsed_times.append(elapsed)
            # logger.debug(f"functioncall took {elapsed:.4f} seconds, header: {header}.)")

        p50 = median(elapsed_times)
        p90 = calculate_percentile(elapsed_times, 90)
        p99 = calculate_percentile(elapsed_times, 99)
        logger.info(
            f"Longest functioncall took {max_elapsed:.4f} seconds, header: {max_elapsed_header}, p50: {p50}, p90: {p90}, p99: {p99}"
        )

        return data_list


def get_function_name(runtime_type):
    if runtime_type == "python_code":
        return "realhf_code_verify"
    elif runtime_type == "python_math":
        return "realhf_math_verify"
    return "empty_code"


def batch_function_call(payload_list, runtime_type, timeout=30):
    start_time = time.time()
    function_name = get_function_name(runtime_type)
    result = asyncio.run(
        batch_function_call_async(payload_list, function_name, timeout)
    )
    execution_time = time.time() - start_time
    logger.debug(
        f"Batch function call done, runtime type: {runtime_type}, batch size: {len(payload_list)}, cost: {execution_time * 1000:.0f} ms"
    )
    return result

from loguru import logger
import asyncio
import aiofiles
import json

from ..utils.draw_metrics import draw_metrics


async def profiler_stop_cleanup(state):
    async with state.data_lock:
        state.is_profiling = False
    # Save profile data and generate report Here
    output_data = {}
    if state.profiling_task:
        try:
            logger.info("Stopping profiling and saving data...")
            output_data = await asyncio.wait_for(state.profiling_task, timeout=5.0)
        except asyncio.TimeoutError:
            # If the task did not finish in time, we log a warning and cancel it
            # No profile data will be saved if the task is not completed
            logger.warning("Profiling task did not finish in time. Data might be incomplete for the last frame.")
            state.profiling_task.cancel()
        except Exception as e:
            logger.error(f"Error during profiling task shutdown: {e}")
            # Print full traceback for debugging
            import traceback

            logger.error(traceback.format_exc())
    else:
        logger.warning("No profiling task was running. No data to save.")

    output_data["request_time_info"] = state.time_consumption_data
    output_data["frames"] = state.profiling_data_frames

    try:
        async with aiofiles.open(state.output_filename, mode="w") as f:
            await f.write(json.dumps(output_data, indent=2))
        logger.info(f"Profiling data successfully saved to {state.output_filename}")

    except Exception as e:
        logger.error(f"Failed to save profiling data: {e}")

    try:
        draw_metrics(output_data, state.report_filename, state.report_metrics)
        logger.info(f"Profiling report generated: {state.report_filename}")
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")

# Copyright © 2023 Apple Inc.

"""SPMD related utils."""

import logging
import time
from typing import Optional
import orbax.checkpoint as ocp

import jax
import portpicker
from etils import epath

_jax_distributed_initialized = False


def setup(
    *,
    jax_backend: str,
    distributed_coordinator: Optional[str] = None,
    num_processes: Optional[int] = None,
    process_id: Optional[int] = None,
    initialization_timeout: Optional[int] = None,
    local_checkpoint_dir: Optional[str] = None,
):
    """Sets up the JAX environment for SPMD.

    Args:
        jax_backend: The distributed backend, which can be "cpu", "gpu", or "tpu".
        distributed_coordinator: The distributed coordinator address (in the form of <host>:<port>).
            Needed only for `jax_backend != "tpu"` and `num_processes > 1`. Otherwise, the
            coordinator will be configured automatically when `num_processes` and `process_id` are
            provided.
        num_processes: The number of processes. Needed only if distributed initialization is desired
            for `jax_backend != "tpu"`.
        process_id: The process ID (the process rank). Needed only if distributed initialization is
            desired for `jax_backend != "tpu"`.
        initialization_timeout: The jax distributed initialization timeout in seconds. If None, uses
            jax default.

    Raises:
        ValueError: If any of the following conditions are met:
            * distributed_coordinator, num_processes, or process_id are not None when
                jax_backend is "tpu";
            * one of num_processes or process_id is None when jax_backend is not "tpu";
            * distributed_coordinator is None when jax_backend is not "tpu" and num_processes > 1.
    """
    # Use a GSPMD-friendly PRNG implementation.
    jax.config.update("jax_default_prng_impl", "rbg")
    # This allows replicated jax.Arrays to be used for computation on the host.
    jax.config.update("jax_spmd_mode", "allow_all")

    global _jax_distributed_initialized  # pylint: disable=global-statement
    if not _jax_distributed_initialized:
        init_kwargs = {}
        if initialization_timeout is not None:
            init_kwargs["initialization_timeout"] = initialization_timeout

        if jax_backend == "tpu":
            if not (
                distributed_coordinator is None and num_processes is None and process_id is None
            ):
                raise ValueError(
                    "distributed_coordinator, num_processes, and process_id "
                    "should all be None for tpu backend."
                )
        else:
            if distributed_coordinator is None and num_processes is None and process_id is None:
                logging.info(
                    "Skipping distributed initialization for %s backend, "
                    "since distributed_coordinator, num_processes, and process_id are all None.",
                    jax_backend,
                )
                return

            if num_processes is None or process_id is None:
                raise ValueError(
                    "num_processes and process_id should be provided together "
                    f"if distributed initialization is desired for backend {jax_backend}. "
                    f"Instead, got num_processes={num_processes}, process_id={process_id}."
                )

            if not distributed_coordinator:
                if num_processes == 1:
                    distributed_coordinator = f"localhost:{portpicker.pick_unused_port()}"
                else:
                    raise ValueError(f"Unknown distributed_coordinator: {distributed_coordinator}")

            init_kwargs.update(
                coordinator_address=distributed_coordinator,
                num_processes=num_processes,
                process_id=process_id,
            )

        # Ensure that coordinator initialization respects initialization_timeout.
        # The current jax version hardcodes the number of attempts to discover coordinator address:
        # https://github.com/google/jax/blob/33e1a96204bf88f76b9f8d982cb2fe92ebcf0151/jax/_src/clusters/cloud_tpu_cluster.py#L110-L125
        # TODO(markblee): Remove this once we update jax to include:
        # https://github.com/google/jax/commit/c5869feb92a175ce40b4e5a897f505e97bcc610b.
        if initialization_timeout is not None:
            max_time = time.time() + initialization_timeout
            while not _jax_distributed_initialized and time.time() < max_time:
                try:
                    logging.info("Attempting to initialize JAX distributed system...")
                    logging.info(f"LOCAL CHECKPOINT DIR = {local_checkpoint_dir}")
                    if local_checkpoint_dir is not None:
                        logging.info("Using Emergency Checkpointing...")
                        initialize_jax_for_tpu_with_emergency_checkpointing(local_checkpoint_dir,**init_kwargs)
                    else:
                        jax.distributed.initialize(**init_kwargs)
                    _jax_distributed_initialized = True
                except RuntimeError as e:
                    err_str = str(e)
                    if (
                        "Failed to recognize coordinator" in err_str
                        or "Barrier timed out" in err_str
                    ):
                        continue  # Retry. Sleep is handled by jax.distributed.initialize.
                    raise
            if not _jax_distributed_initialized:
                raise RuntimeError(
                    "Failed to initialize JAX distributed system within the specified timeout "
                    f"({initialization_timeout} seconds)."
                )
        else:
            if local_checkpoint_dir is not None:
                logging.info("2: Using Emergency Checkpointing."
                            "Attempting to initialize JAX distributed system...")
                initialize_jax_for_tpu_with_emergency_checkpointing(local_checkpoint_dir,**init_kwargs)
            else:
                jax.distributed.initialize(**init_kwargs)
            _jax_distributed_initialized = True

def initialize_jax_for_tpu_with_emergency_checkpointing(local_checkpoint_dir,**init_kwargs):
    """Initialize JAX distributed runtime for TPUs when emergency checkpointing is used.
    The information required to initialize JAX distributed runtime will be written by GKE to
    the local checkpoint directory. This function retrieves that information and initializes
    JAX distributed runtime.
    """
    process_id, coordinator_address = _retrieve_jax_init_info(local_checkpoint_dir)

    if process_id != "" and coordinator_address != "":
        logging.info(f"Using {process_id} as the process_id and {coordinator_address} as the"
                        " coordinator_address to initialize JAX distributed runtime...")
        jax.distributed.initialize(coordinator_address=coordinator_address, process_id=int(process_id))
    else:
        logging.info("Initializing JAX distributed runtime without args when emergency checkpointing is"
                        " enabled. This should not happen and your workload may have unexpected behavior.")
        jax.distributed.initialize(**init_kwargs)

    ocp.multihost.utils.initialize_runtime_to_distributed_ids()

def _retrieve_jax_init_info(local_checkpoint_dir):
    """Retrieve JAX init info from a local file."""
    JAX_INIT_INFO_FILE = "jax-init-info.txt"
    local_jax_init_info_file = epath.Path(local_checkpoint_dir) / JAX_INIT_INFO_FILE
    # Allow time for the JAX init info file to be populated by GKE. This is needed because the file is
    # only populated when the worker with process id of 0 is determined. After a disruption, although some
    # workers might be up and running, the init info file won't be populated until the node with process id
    # of 0 is known and this could take time. Using 900 seconds for now and it needs to be increased if the
    # "repair" time is longer.
    for i in range(900):
        if local_jax_init_info_file.exists():
            return local_jax_init_info_file.read_text().split('\n')[:2]
        logging.info(f"Unable to locate {JAX_INIT_INFO_FILE} after {i} seconds, sleeping for 1 second before retrying...")
        time.sleep(1)
    logging.info(f"Unable to locate {JAX_INIT_INFO_FILE} after 900 seconds,"
                    "returning empty process id and coordinator address.")
    return "", ""
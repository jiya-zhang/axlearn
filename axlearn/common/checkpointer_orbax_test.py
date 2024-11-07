# Copyright © 2024 Apple Inc.

"""Tests orbax checkpointer.

See also checkpointer_test.py for common checkpointing tests.
"""

# pylint: disable=protected-access

import logging
import multiprocessing as mp
import os
import socket
import tempfile
from contextlib import ExitStack, closing
from typing import Sequence

import jax
import numpy as np
import orbax.checkpoint as ocp
import pytest
import tensorflow as tf
from absl.logging import PythonFormatter
from jax import numpy as jnp
from jax.experimental import mesh_utils

from axlearn.common import test_utils
from axlearn.common.checkpointer import read_index_file
from axlearn.common.checkpointer_orbax import (
    OrbaxCheckpointer,
    OrbaxEmergencyCheckpointer,
    config_for_function,
    every_n_steps_policy,
    ocp,
)


def _find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _mesh(mesh_shape: Sequence[int]):
    devices = mesh_utils.create_device_mesh(mesh_shape)
    return jax.sharding.Mesh(devices, ("data", "model"))


def _logger_init():
    handler = logging.StreamHandler()
    handler.setFormatter(PythonFormatter())
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)


def _test_orbax_main(process_id: int, port: int, persist_dir: str, local_dir: str, q: mp.Queue):
    # pylint: disable=import-outside-toplevel
    from orbax.checkpoint._src.multihost import multislice
    from orbax.checkpoint.experimental.emergency import checkpoint_manager

    # Patch for GPU use. We don't need to use mock.patch because we're running in a subprocess.
    multislice.get_device_memory = lambda: int(80e9)

    def slice_devices(
        global_mesh: jax.sharding.Mesh,
        *,
        replica_id: int = 0,
        replica_axis_index: int = 0,
    ) -> np.ndarray:
        return np.take(
            global_mesh.devices,
            replica_id,
            axis=replica_axis_index,
        )

    def _all_devices_excepting_slice(
        devices: np.ndarray,
        *,
        replica_id: int = 0,
        replica_axis_index: int = 0,
    ) -> np.ndarray:
        return np.delete(devices, replica_id, axis=replica_axis_index)

    # We're not running in a true multi-slice environment. Patch the following two functions to
    # mock multi-slice discovery.
    multislice.slice_devices = slice_devices
    checkpoint_manager._all_devices_excepting_slice = _all_devices_excepting_slice

    jax.distributed.initialize(
        coordinator_address=f"127.0.0.1:{port}",
        num_processes=4,
        process_id=process_id,
        local_device_ids=[process_id],
    )

    cfg: OrbaxEmergencyCheckpointer.Config = OrbaxEmergencyCheckpointer.default_config()
    cfg.name = "emergency"
    cfg.save_policy = config_for_function(every_n_steps_policy).set(n=25)
    cfg.local_save_policy = config_for_function(every_n_steps_policy).set(n=5)
    # Local checkpoint path suffix must be the same for orbax synchronization to work.
    cfg.local_dir = os.path.join(local_dir, "checkpoints")
    cfg.dir = persist_dir
    cfg.keep_last_n = 2
    cfg.replica_axis_index = 0
    cfg.async_timeout_secs = 5
    cfg.non_tensor_async_timeout_secs = 5
    checkpointer: OrbaxEmergencyCheckpointer = cfg.instantiate(parent=None)
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(2, 2), ["data", "model"])
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "model"))
    x = jax.make_array_from_process_local_data(
        sharding, np.zeros((1024, 1024), dtype=np.float32), (1024, 1024)
    )

    input_iter = iter(tf.data.Dataset.counter())
    state = {"x": x, "y": None, "z": input_iter}
    with mesh:
        step, state = checkpointer.restore(step=None, state=state)
        if step is None:
            step = 0
        else:
            # This is the step of the last persistent ckpt.
            assert checkpointer.latest_checkpoint_step(checkpointer.config.dir) == 25
            # This step should already be garbage collected.
            assert 20 not in checkpointer._tensor_manager.all_steps()
            # Although the persistent save interval is 25 steps, local save interval is 5
            # steps, so we should be able to restore step 40.
            assert step >= 40
            # Since we save after step X is finished, after restore the first step is X + 1.
            step += 1
        for i in range(step, 100):
            assert jnp.all(state["x"] == i).item()
            assert i == next(input_iter)  # input_iter's state is modified inplace.
            state = {"x": state["x"] + 1, "y": state["y"], "z": state["z"]}
            checkpointer.save(step=i, state=state)
            if process_id == 0:
                logging.info("step %d", i)
            if i == 45 and q is not None and process_id == 0:
                # Signal processes to be killed at around step 45.
                q.put(1)

        checkpointer.wait_until_finished()
    jax.distributed.shutdown()


class OrbaxCheckpointerTest(test_utils.TestCase):
    def test_index(self):
        """Tests that index files saved with orbax can be read with `read_index_file`."""
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape), tempfile.TemporaryDirectory() as temp_dir:
            ckpt = (
                OrbaxCheckpointer.default_config()
                .set(name="test", dir=temp_dir)
                .instantiate(parent=None)
            )
            step = 123
            state = dict(x=jnp.ones([3, 2]))
            ckpt.save(step=step, state=state)
            ckpt.wait_until_finished()

            ref_index = read_index_file(os.path.join(temp_dir, "step_00000123", "index"))
            test_index = ckpt._manager.restore(
                step=step,
                # The input iterator is saved as part of `save_tf_savables`.
                args=ocp.args.Composite(
                    index=ocp.args.JsonSave(ckpt._get_spec(step=step, state=state))
                ),
            )
            self.assertEqual(ref_index, test_index["index"])

    # This test requires 4 devices to run. Note: we cannot use skipif(jax.local_device_count() < 4)
    # because it will initialize the backend, causing the jax.distributed.initialize to fail in
    # _test_orbax_main. Using the `spawn` context from multiprocessing results in a different error.
    # Since we don't have GPU/TPU CI anyway, we always skip this test.
    @pytest.mark.skipif(True, reason="This test needs to be run manually.")
    def test_emergency_ckpt(self):
        with ExitStack() as stack:
            num_processes = 4
            local_tempdirs = [
                stack.enter_context(tempfile.TemporaryDirectory()) for _ in range(num_processes)
            ]
            persistent_tempdir = stack.enter_context(tempfile.TemporaryDirectory())
            q = mp.Queue()
            # Populate log messages.
            _logger_init()

            def start_processes() -> list[mp.Process]:
                free_port = _find_free_port()
                processes = []
                for i in range(num_processes):
                    p = mp.Process(
                        target=_test_orbax_main,
                        args=(
                            i,
                            free_port,
                            persistent_tempdir,
                            local_tempdirs[i],
                            None if i > 0 else q,
                        ),
                    )
                    processes.append(p)
                    p.start()
                return processes

            processes = start_processes()

            # Block until we get a signal from a process.
            q.get()

            # Kill all processes to simulate a failure.
            for p in processes:
                p.kill()

            processes = start_processes()

            try:
                for p in processes:
                    p.join()
                for p in processes:
                    self.assertEqual(p.exitcode, 0)
            finally:
                for p in processes:
                    p.kill()

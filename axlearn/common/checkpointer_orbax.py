# Copyright © 2024 Apple Inc.

"""Checkpointing utilities using orbax.

See also checkpointer.py for other checkpointing utilities and checkpointer_test.py for tests.
"""

import asyncio
import copy
import dataclasses
import functools
import os
import time
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.experimental.emergency.checkpoint_manager as oecp
import tensorflow as tf
from absl import logging
from jax._src.mesh import thread_resources
from jax.experimental.array_serialization import serialization

# TODO(hanzhi-zhou): fix this after orbax move metadata back to public API.
from orbax.checkpoint._src.metadata.value import Metadata
from orbax.checkpoint._src.multihost import multihost as multihost_src

from axlearn.common import file_system as fs
from axlearn.common import utils
from axlearn.common.checkpointer import (
    STEP_NUM_DIGITS,
    STEP_PREFIX,
    BaseCheckpointer,
    Checkpointer,
    CheckpointPolicy,
    CheckpointValidationType,
    InstantiableConfig,
    StateStorage,
    StateStorageCommitCallback,
    async_save_tf_savables,
    check_state_structure,
    config_for_function,
    every_n_steps_policy,
    maybe_restore_grain_savables,
    maybe_save_grain_savables,
    multihost_utils,
    parse_step_from_dir,
    read_index_file,
    restore_tf_savables,
    write_index_file,
)
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.module import Module
from axlearn.common.utils import Nested, Tensor, TensorSpec
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
from orbax.checkpoint._src.metadata import value as metadata

try:
    # The import also registers the checkpoint handlers.
    import grain.python as grain

    _GrainIterator = Union[grain.DatasetIterator, grain.PyGrainDatasetIterator]
    _GRAIN_INSTALLED = True
except ImportError:
    logging.warning("grain is not installed; checkpointing grain iterators will not work.")
    _GRAIN_INSTALLED = False


class _TfIteratorHandler(ocp.type_handlers.TypeHandler):
    """Serializes tf.data.Iterator.

    Reference:
    https://orbax.readthedocs.io/en/latest/custom_handlers.html#custom-serialization-deserialization

    Note that since handlers are global instances (potentially shared by multiple checkpointer
    instances), we construct and cleanup the executor per-serialize/deserialize call.
    """

    # Must be a subclass of RestoreArgs for `PyTreeRestore` to recognize it.
    @dataclasses.dataclass
    class RestoreArgs(ocp.type_handlers.RestoreArgs):
        item: Optional[tf.data.Iterator] = None

    def typestr(self) -> str:
        return "TfIterator"

    async def serialize(
        self,
        values: Sequence[tf.data.Iterator],
        infos: Sequence[ocp.type_handlers.ParamInfo],
        args: Optional[Sequence[ocp.args.PyTreeSave]],
    ) -> List[futures.Future]:
        """Serializes `values` into corresponding `info.path`s."""
        del args  # Unused.
        futs = []
        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            for value, info in zip(values, infos):
                futs.append(async_save_tf_savables(value, executor=executor, dir=info.path))
        return futs

    async def deserialize(
        self,
        infos: Sequence[ocp.type_handlers.ParamInfo],
        args: Optional[Sequence[RestoreArgs]] = None,
    ) -> Sequence[tf.data.Iterator]:
        if args is None:
            raise ValueError(f"{self.RestoreArgs.__name__} should be supplied as args.")
        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            futs = [
                asyncio.get_event_loop().run_in_executor(
                    executor,
                    functools.partial(restore_tf_savables, arg.item, dir=info.path),
                )
                for arg, info in zip(args, infos)
            ]
        return await asyncio.gather(*futs)

    async def metadata(self, infos: Sequence[ocp.type_handlers.ParamInfo]) -> Sequence[Metadata]:
        return [Metadata(name=info.name, directory=info.path) for info in infos]


ocp.type_handlers.register_type_handler(tf.data.Iterator, _TfIteratorHandler(), override=True)


if _GRAIN_INSTALLED:

    class _GrainDatasetIteratorHandler(ocp.type_handlers.TypeHandler):
        """Serializes grain dataset iterators."""

        @dataclasses.dataclass
        class RestoreArgs(ocp.type_handlers.RestoreArgs):
            item: Optional[_GrainIterator] = None

        def typestr(self) -> str:
            return "DatasetIterator"

        async def serialize(
            self,
            values: Sequence[grain.DatasetIterator],
            infos: Sequence[ocp.type_handlers.ParamInfo],
            args: Optional[Sequence[ocp.args.PyTreeSave]],
        ) -> List[futures.Future]:
            """Serializes `values` into corresponding `info.path`s."""
            del args  # Unused.
            for value, info in zip(values, infos):
                ckpt_dir = os.path.dirname(info.path)
                path = os.path.basename(info.path)
                maybe_save_grain_savables({path: value}, dir=ckpt_dir)
            return []

        async def deserialize(
            self,
            infos: Sequence[ocp.type_handlers.ParamInfo],
            args: Optional[Sequence[RestoreArgs]] = None,
        ) -> Sequence[_GrainIterator]:
            if args is None:
                raise ValueError(f"{self.RestoreArgs.__name__} should be supplied as args.")
            return [
                maybe_restore_grain_savables(arg.item, dir=info.path)
                for arg, info in zip(args, infos)
            ]

        async def metadata(
            self, infos: Sequence[ocp.type_handlers.ParamInfo]
        ) -> Sequence[Metadata]:
            return [Metadata(name=info.name, directory=info.path) for info in infos]

    ocp.type_handlers.register_type_handler(
        grain.DatasetIterator, _GrainDatasetIteratorHandler(), override=True
    )


class OrbaxCheckpointer(BaseCheckpointer):
    """A checkpointer that uses orbax CheckpointManager.

    NOTE: While this class uses index files to do additional validation on checkpoint state, the
    index file is not used for committing checkpoints (i.e., it is not reliable to check for the
    presence of 'index' file). In Orbax, checkpoints are committed differently on different
    filesystems. If the filesystem supports atomic renames, un-committed checkpoints are represented
    by a "tmp" prefix. OTOH, if the filesystem does not support atomic renames (such as GCS as of
    writing), committed checkpoints are represented by a commit file with the name
    "commit_success.txt". Thus, an incomplete checkpoint can still contain an "index" file (note
    that valid checkpoints should always contain an "index" file). In general, users should instead
    use `checkpoint_paths` to identify committed checkpoints.
    """

    @config_class
    class Config(BaseCheckpointer.Config):
        """Configures OrbaxCheckpointer.

        Attributes:
            keep_last_n: Keep this many past ckpts.
            validation_type: Checkpoint validation during restore.
            async_timeout_secs: Timeout for async barrier in seconds.
        """

        keep_last_n: int = 1
        validation_type: CheckpointValidationType = CheckpointValidationType.EXACT
        async_timeout_secs: int = 300

    @classmethod
    def checkpoint_paths(cls, base_dir: str) -> List[str]:
        """See `BaseCheckpointer.checkpointer_paths`."""
        logging.log_first_n(
            logging.WARNING,
            msg="checkpoint_paths is deprecated. Use checkpoint_steps instead.",
            n=1,
        )
        return [str(path) for path in ocp.utils.checkpoint_steps_paths(base_dir)]

    @classmethod
    def checkpoint_steps(cls, base_dir) -> list[int]:
        """See `BaseCheckpointer.checkpointer_steps`."""
        return ocp.utils.checkpoint_steps(base_dir)

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)

        cfg: OrbaxCheckpointer.Config = self.config
        save_policy = cfg.save_policy.instantiate()

        # self._eval_summaries will be set in save() and used by save_fn_with_summaries() to decide
        # whether to save at the step.
        #
        # We need this because Orbax only provides (current_step, last_saved_step) as args to the
        # save policy, so we use `self._eval_summaries` to capture evaler summaries.
        #
        # While we can check the eval summaries in `save()`, some Orbax features like
        # save-on-preemption requires the user to call `ocp.CheckpointManager.save()` at every step,
        # even if the verdict from `cfg.save_policy` is negative.
        self._eval_summaries = None

        def save_fn_with_summaries(step: int, last_saved_step: Optional[int]) -> bool:
            del last_saved_step
            return save_policy(step=step, evaler_summaries=self._eval_summaries)

        self._name_format = ocp.step.standard_name_format(
            step_prefix=STEP_PREFIX,
            step_format_fixed_length=STEP_NUM_DIGITS,
        )
        self._manager = ocp.CheckpointManager(
            directory=cfg.dir,
            options=ocp.CheckpointManagerOptions(
                create=True,
                max_to_keep=cfg.keep_last_n,
                enable_async_checkpointing=True,
                step_name_format=self._name_format,
                should_save_fn=save_fn_with_summaries,
                enable_background_delete=True,
                async_options=ocp.options.AsyncOptions(timeout_secs=cfg.async_timeout_secs),
            ),
            item_handlers={
                # NOTE: we make a relatively weak assumption that index files are JSON serialized
                # for simplicity. The test cases ensure that this is compatible with
                # `read_index_file`.
                "index": ocp.JsonCheckpointHandler(filename="index"),
                # TODO(markblee): Add save/restore_concurrent_gb when available.
                # Note that this defaults to use_ocdb=True. Note also that custom `TypeHandler`s are
                # ignored by `StandardCheckpointHandler`, so we use `PyTreeCheckpointHandler`.
                "state": ocp.PyTreeCheckpointHandler(),
            },
        )

    def _get_spec(self, *, step: int, state: Nested[Any]) -> Nested[Any]:
        spec = {"index": [("step", step)]}
        for path, value in utils.flatten_items(state):
            if isinstance(value, (Tensor, TensorSpec)):
                dtype = getattr(value.dtype, "dtype", value.dtype)
                spec["index"].append(
                    (path, {"dtype": str(dtype), "shape": str(tuple(value.shape))})
                )
            elif (
                isinstance(value, tf.data.Iterator)
                or _GRAIN_INSTALLED
                and isinstance(value, _GrainIterator)
            ):
                spec["index"].append((path, str(type(value))))
            else:
                spec["index"].append((path, value))
        return spec

    # pylint: disable-next=redefined-builtin
    def ckpt_dir(self, step: int, dir: Optional[str] = None) -> str:
        """Obtains the checkpoint dir for the given step."""
        if dir is None:
            dir = self._manager.directory
        return str(ocp.step.build_step_path(dir, self._name_format, step))

    def save(
        self, *, step: int, state: Nested[Tensor], evaler_summaries: Optional[Dict[str, Any]] = None
    ):
        """See `BaseCheckpointer.save` for details.

        Checkpoint saving is handled by `orbax` checkpoint manager.
        """
        spec = self._get_spec(step=step, state=state)
        assert self._eval_summaries is None, self._eval_summaries
        self._eval_summaries = copy.deepcopy(evaler_summaries or {})

        try:
            # Note that save() waits for prior serialization to finish.
            self._manager.save(
                step=step,
                # The input iterator is saved as part of `save_tf_savables`.
                args=ocp.args.Composite(
                    index=ocp.args.JsonSave(spec["index"]),
                    # TODO(markblee): Investigate save_args for chunk_byte_size and
                    # ocdbt_target_data_file_size:
                    # https://orbax.readthedocs.io/en/latest/optimized_checkpointing.html#custom-chunk-sizes
                    # https://orbax.readthedocs.io/en/latest/optimized_checkpointing.html#customizing-data-file-size
                    state=ocp.args.PyTreeSave(item=state),
                ),
            )
            # Exit early after pre-emption, equivalent to sys.exit():
            # https://orbax.readthedocs.io/en/latest/preemption_checkpointing.html
            if self._manager.reached_preemption(step):
                self._manager.wait_until_finished()
                raise SystemExit(f"Exiting after saving checkpoint at {step=} due to pre-emption.")
        finally:
            self._eval_summaries = None

    def restore(
        self,
        *,
        step: Optional[int] = None,
        state: Union[Nested[Tensor], Nested[TensorSpec]],
    ) -> Tuple[Optional[int], Nested[Tensor]]:
        """See `BaseCheckpointer.restore` for details."""

        cfg: OrbaxCheckpointer.Config = self.config

        def _restore_args(x: Any) -> ocp.RestoreArgs:
            if isinstance(x, (Tensor, TensorSpec)):
                return ocp.checkpoint_utils.construct_restore_args(
                    jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding)
                )
            elif isinstance(x, tf.data.Iterator):
                return _TfIteratorHandler.RestoreArgs(item=x)
            elif _GRAIN_INSTALLED and isinstance(x, _GrainIterator):
                return _GrainDatasetIteratorHandler.RestoreArgs(item=x)
            else:
                return None

        restore_args = jax.tree.map(_restore_args, state)

        try:
            composite_state = self._manager.restore(
                step,
                args=ocp.args.Composite(
                    index=ocp.args.JsonRestore(None),
                    state=ocp.args.PyTreeRestore(item=state, restore_args=restore_args),
                ),
            )
        except TypeError as e:
            # Orbax hits TypeError if there are no checkpoints, since it attempts to format `None`
            # as the step dir.
            if step is not None:
                raise ValueError(f"Failed to restore at step {step}.") from e
            logging.info("Could not find any completed checkpoints under %s: %s", cfg.dir, e)
            return None, state  # Return the input state.

        restored_index = composite_state["index"]
        restored_state = composite_state["state"]

        # If we successfully restored from step=None, use the restored step.
        if step is None:
            for k, v in restored_index:
                if k == "step":
                    step = v
                    break

        # Validate ckpt structure.
        check_state_structure(
            restored_index,
            target_structure=self._get_spec(step=step, state=state)["index"],
            validation=cfg.validation_type,
        )
        return step, restored_state

    def wait_until_finished(self):
        """See `BaseCheckpointer.wait_until_finished` docstring for details."""
        self._manager.wait_until_finished()

    def stop(self):
        """See `BaseCheckpointer.stop` for details."""
        self._manager.close()


class _TFSavablesStateStorage(StateStorage):
    """A StateStorage implementation that only saves the index file and tf savables."""

    @config_class
    class Config(StateStorage.Config):
        timeout_secs: int = 300

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # One thread is sufficient because `async_save_tf_savables` only creates one future.
        self._executor = ThreadPoolExecutor(1)
        self._manager = serialization.AsyncManager(timeout_secs=cfg.timeout_secs)

    def _get_spec(self, *, step: int, state: Nested[Any]) -> Nested[Any]:
        spec = {"index": [("step", int(step))], "tf_ckpt_map": {}}
        for path, value in utils.flatten_items(state):
            if isinstance(value, (Tensor, TensorSpec)):
                dtype = getattr(value.dtype, "dtype", value.dtype)
                spec["index"].append(
                    (path, {"dtype": str(dtype), "shape": str(tuple(value.shape))})
                )
            elif isinstance(value, tf.data.Iterator):
                spec["index"].append((path, str(type(value))))
                spec["tf_ckpt_map"][path] = value
            else:
                spec["index"].append((path, value))
        logging.log_first_n(logging.INFO, "TF savables spec: %s", 1, str(spec))
        return spec

    def save_to_dir(
        self,
        *,
        step: int,
        state: Nested[Tensor],
        ckpt_dir: str,
        on_commit_callback: StateStorageCommitCallback = write_index_file,
    ):
        start_time = time.perf_counter()
        # We write data files directly to `ckpt_dir`. `index` is written into `ckpt_dir` in
        # `on_commit_callback` to finalize the checkpoint.
        spec = self._get_spec(step=step, state=state)
        self.wait_until_finished()

        save_tf_future = async_save_tf_savables(
            spec["tf_ckpt_map"],
            executor=self._executor,
            dir=os.path.join(ckpt_dir, f"tf_{jax.process_index()}"),
        )

        def commit():
            on_commit_callback(ckpt_dir=ckpt_dir, index=spec["index"])
            logging.info(
                "Serialization of TF savables to %s completed in %s seconds.",
                ckpt_dir,
                time.perf_counter() - start_time,
            )

        # pylint: disable=protected-access
        self._manager._add_futures([save_tf_future])
        self._manager._start_async_commit(commit)

    def wait_until_finished(self):
        self._manager.wait_until_finished()

    def restore_from_dir(
        self,
        step: int,
        state: Union[Nested[Tensor], Nested[TensorSpec]],
        *,
        ckpt_dir: str,
        validation: CheckpointValidationType = CheckpointValidationType.EXACT,
    ) -> Nested[Tensor]:
        spec = self._get_spec(step=step, state=state)
        logging.info("Restoring TF savables from directory %s", ckpt_dir)
        check_state_structure(
            read_index_file(ckpt_dir), target_structure=spec["index"], validation=validation
        )
        restore_tf_savables(
            spec["tf_ckpt_map"], dir=os.path.join(ckpt_dir, f"tf_{jax.process_index()}")
        )
        multihost_utils.sync_global_devices(ckpt_dir)
        return state

    def stop(self):
        self._executor.shutdown(wait=True)


def _initialize_runtime_to_distributed_ids(timeout: int):
    """Initializes orbax's internal process index mapping.

    This function is ported from orbax's source code to make the timeout configurable.
    https://github.com/google/orbax/blob/073880cae248fc721fe31e46bf1bb386346d3aa5/checkpoint/orbax/checkpoint/_src/multihost/multihost.py#L52
    """
    # pylint: disable=protected-access
    client = multihost_src._get_jax_distributed_client()

    # Index is distributed id.
    # Value is runtime id.
    multihost_src._RUNTIME_TO_DISTRIBUTED_ID = [0 for _ in range(jax.process_count())]
    own_runtime_id = jax.process_index()
    own_distributed_id = (
        jax._src.distributed.global_state.process_id  # pytype: disable=module-attr
    )  # pylint: disable=protected-access
    dir_key = "jax/process_id/"
    key = dir_key + str(own_runtime_id)
    client.key_value_set(key, str(own_distributed_id))
    client.wait_at_barrier("orbax_global_discovery", timeout_in_ms=timeout * 1000)
    ids = client.key_value_dir_get(dir_key)
    for key, distributed_id in ids:
        runtime_id = int(key.split("/")[-1])
        multihost_src._RUNTIME_TO_DISTRIBUTED_ID[runtime_id] = int(distributed_id)
    logging.info(
        "[process=%s][thread=%s] runtime_to_distributed_id: %s",
        multihost_src.process_index(),
        multihost_src.threading.current_thread().name,
        multihost_src._RUNTIME_TO_DISTRIBUTED_ID,
    )


class OrbaxEmergencyCheckpointer(BaseCheckpointer):
    """Checkpointer implementation that uses Orbax emergency checkpoint.

    This checkpointer is intended for multi-slice training that uses data-parallelism across
    slices. Orbax emergency checkpoint works by exploiting the following properties
    1.  Tensors are replicated across data-parallel replicas.
    2.  When a slice fails in a multi-slice training and failover is started, only nodes
        corresponding to the non-healthy slice may be restarted. Healthy nodes from healthy slices
        will not restart.

    Hence, all slices can write checkpoints to node's memory or disk, providing us with redundancy
    when there's a failure. This checkpoint frequency can be much higher than remote filesystem,
    which has limited bandwidth to support high frequency saving. Checkpoints on nodes are referred
    as local checkpoints. Checkpoints on remote filesystem are referred as persistent checkpoints.

    When a failure occurs, Orbax checkpointer will find the latest step from all local and
    persistent checkpoints. If the checkpoint is local, the slice on which that checkpoint is
    stored will read the checkpoint and broadcast the read values to other slices.

    However, the above procedure doesn't apply to some non-tensor states such as data iterators.
    Data iterators are unique across jax processes, and thus cannot be stored on nodes. Orbax
    emergency checkpointer doesn't support non-tensor states. Therefore, we reuse axlearn
    Checkpointer to save, restore and garbage collect those states, which include the index file
    and tf iterators. These non-tensor states will be saved whenever local or persistent checkpoint
    need to be saved. As the result, the persistent checkpoint structure looks like this:

    ├── path_prefix
    │   ├── non-tensors
    │   │   └── step_00000010
    │   │       ├── index
    │   │       └── tf_xxx
    │   └── tensors
    │       └── step_00000010
    │           └── orbax_files_xxx

    A persistent training checkpoint `step_xxx` is commited when `non-tensors/step_xxx/index`
    exists and `tensors/step_xxx` is commited by Orbax. Refer to the docstring of
    `OrbaxCheckpointer` for Orbax's commit criteria.

    To abstract the details of the checkpoint layout, the `checkpoint_steps` API returns all steps
    for which both Tensor and non-Tensor states have been fully committed.
    """

    _NON_TENSORS_PREFIX: str = "non-tensors"
    _TENSORS_PREFIX: str = "tensors"

    @config_class
    class Config(BaseCheckpointer.Config):
        """Configures OrbaxEmergencyCheckpointer.

        Attributes:
            keep_last_n: Keep this many past ckpts.
            keep_every_n_steps: If > 0, keeps at least one persistent checkpoint every N steps.
            local_keep_last_n: Keep this many past ckpts in local storage (e.g. node memory).
                This should almost always set to 1 to avoid OOM.
            local_dir: Ckpt path for local storage. The content in this path must persist across
                pod restarts unless the restart is caused by node failure. `local_dir` must be the
                same for all processes or processes may hang.
            save_policy: Save policy for persistent checkpoints.
            local_save_policy: Save policy for local checkpoints. This should be more frequent than
                `save_policy`. Note that data iterator will be saved with either `save_policy` or
                `local_save_policy` indicate we should save.
            non_tensor_async_timeout_secs: Timeout for async barrier in seconds when saving
                non-tensor states.
            async_timeout_secs: Timeout for async barrier in seconds when saving tensors.
            replica_axis_index: The index of the "data" axis.
        """

        keep_last_n: int = 1
        keep_every_n_steps: Optional[int] = None
        local_keep_last_n: int = 1
        local_save_policy: InstantiableConfig[CheckpointPolicy] = config_for_function(
            every_n_steps_policy
        ).set(n=10)
        local_dir: str = "/host-tmp/checkpoints"
        non_tensor_async_timeout_secs: int = 300
        async_timeout_secs: int = 3600
        replica_axis_index: Required[int] = REQUIRED

    @classmethod
    def checkpoint_paths(cls, base_dir: str) -> List[str]:
        """See `BaseCheckpointer.checkpointer_paths`.

        Only persistent checkpoint paths are returned. There's no guarantee that the paths returned
        have committed TF savables. Use `checkpoint_steps` to get steps with both tensors and
        committed TF savables.
        """
        logging.log_first_n(
            logging.WARNING,
            msg="checkpoint_paths is deprecated. Use checkpoint_steps instead.",
            n=1,
        )
        tensors_dir = os.path.join(base_dir, cls._TENSORS_PREFIX)
        return [str(path) for path in ocp.utils.checkpoint_steps_paths(tensors_dir)]

    @classmethod
    def checkpoint_steps(cls, base_dir) -> list[int]:
        """See `BaseCheckpointer.checkpointer_steps`.

        Only persistent checkpoint steps are returned.
        """
        return list(
            set(
                ocp.utils.checkpoint_steps(os.path.join(base_dir, cls._TENSORS_PREFIX))
            ).intersection(
                set(Checkpointer.checkpoint_steps(os.path.join(base_dir, cls._NON_TENSORS_PREFIX)))
            )
        )

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg: OrbaxEmergencyCheckpointer.Config = self.config
        self._name_format = ocp.step.standard_name_format(
            step_prefix=STEP_PREFIX,
            step_format_fixed_length=STEP_NUM_DIGITS,
        )
        if jax.process_index() == 0:
            fs.makedirs(os.path.join(cfg.dir, self._NON_TENSORS_PREFIX))
            fs.makedirs(os.path.join(cfg.dir, self._TENSORS_PREFIX))
        fs.makedirs(cfg.local_dir)
        ocp.multihost.sync_global_processes(
            "axlearn-persistent-dir-create", timeout=cfg.non_tensor_async_timeout_secs
        )
        # Orbax emergency ckpt requires this function to be called prior to checkpointer
        # operations.
        _initialize_runtime_to_distributed_ids(cfg.non_tensor_async_timeout_secs)
        ckpt_cfg: Checkpointer.Config = Checkpointer.default_config()
        # TODO(hanzhi-zhou): this `keep_last_n` may not be what users expect since non-tensor
        # states will save when either local or persistent checkpoint will save.
        ckpt_cfg.keep_last_n = cfg.keep_last_n
        ckpt_cfg.keep_every_n_steps = cfg.keep_every_n_steps
        ckpt_cfg.storage = _TFSavablesStateStorage.default_config()
        ckpt_cfg.storage.timeout_secs = cfg.non_tensor_async_timeout_secs
        ckpt_cfg.dir = os.path.join(cfg.dir, self._NON_TENSORS_PREFIX)
        ckpt_cfg.name = "non-tensors-checkpointer"

        save_policy = cfg.save_policy.instantiate()
        local_save_policy = cfg.local_save_policy.instantiate()

        # Non-tensor states must save when either local or persistent ckpt needs to be saved in
        # order for restore from either to succeed.
        def _composite_save_policy(*, step: int, evaler_summaries: dict[str, Any]):
            return save_policy(step=step, evaler_summaries=evaler_summaries) or local_save_policy(
                step=step, evaler_summaries=evaler_summaries
            )

        ckpt_cfg.save_policy = config_for_function(lambda: _composite_save_policy)
        self._non_tensor_manager: Checkpointer = ckpt_cfg.instantiate(parent=self)
        self._tensor_manager: Optional[oecp.CheckpointManager] = None
        # See comments of _eval_summaries in `OrbaxCheckpointer`.
        self._eval_summaries = None

    # pylint: disable-next=redefined-builtin
    def ckpt_dir(self, step: int, dir: Optional[str] = None) -> str:
        """Obtains the checkpoint dir for the given step."""
        if dir is None:
            dir = self._non_tensor_manager.directory
        return str(ocp.step.build_step_path(dir, self._name_format, step))

    def _get_abstract_state(
        self, state_with_tensors: Nested[Tensor]
    ) -> Nested[jax.ShapeDtypeStruct]:
        """Generate the abstract states required by the Orbax emergency checkpointer."""
        return jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding),
            state_with_tensors,
        )

    def _get_tensor_manager(self, state_with_tensors: Nested[Tensor]) -> oecp.CheckpointManager:
        """Creates the emergency checkpoint manager if not exists.

        We defer the creation of this checkpoint manager because it requires the state dict,
        which is not present during __init__.
        """
        cfg: OrbaxEmergencyCheckpointer.Config = self.config
        if self._tensor_manager is not None:
            return self._tensor_manager

        save_policy = cfg.save_policy.instantiate()
        local_save_policy = cfg.local_save_policy.instantiate()

        def _orbax_save_fn(
            step: int, last_saved_step: Optional[int], wrapped_save_policy: CheckpointPolicy
        ) -> bool:
            del last_saved_step
            return wrapped_save_policy(step=step, evaler_summaries=self._eval_summaries)

        # For meaning of these options, refer to
        # https://github.com/google/orbax/blob/95be2c021bc8cbf4badd83a053ff57b7a9f9b314/checkpoint/orbax/checkpoint/experimental/emergency/checkpoint_manager.py#L277
        self._tensor_manager = oecp.CheckpointManager(
            cfg.local_dir,
            persistent_directory=os.path.join(cfg.dir, self._TENSORS_PREFIX),
            global_mesh=thread_resources.env.physical_mesh,
            abstract_state=self._get_abstract_state(state_with_tensors),
            options=oecp.CheckpointManagerOptions(
                local=oecp.LocalCheckpointOptions(
                    should_save_fn=functools.partial(
                        _orbax_save_fn, wrapped_save_policy=local_save_policy
                    ),
                    max_to_keep=cfg.local_keep_last_n,
                ),
                persistent=oecp.PersistentCheckpointOptions(
                    should_save_fn=functools.partial(
                        _orbax_save_fn, wrapped_save_policy=save_policy
                    ),
                    max_to_keep=cfg.keep_last_n,
                ),
                replica_axis_index=cfg.replica_axis_index,
                async_options=oecp.checkpoint_manager.AsyncOptions(
                    timeout_secs=cfg.async_timeout_secs
                ),
                step_name_format=self._name_format,
                cleanup_tmp_directories=True,
                enable_async_checkpointing=True,
            ),
        )
        return self._tensor_manager

    def save(
        self, *, step: int, state: Nested[Tensor], evaler_summaries: Optional[Dict[str, Any]] = None
    ):
        """See `BaseCheckpointer.save` for details."""
        assert self._eval_summaries is None, self._eval_summaries
        self._eval_summaries = copy.deepcopy(evaler_summaries or {})

        start_t = time.perf_counter()
        state_with_tensors = jax.tree.map(
            lambda x: x if isinstance(x, (Tensor, TensorSpec)) else None, state
        )
        # Note that save() waits for prior serialization to finish.
        self._non_tensor_manager.save(step=step, state=state)
        self._get_tensor_manager(state_with_tensors).save(
            step=step, args=ocp.args.PyTreeSave(item=state_with_tensors)
        )
        self._eval_summaries = None
        if (time_diff := time.perf_counter() - start_t) > 0.5:
            logging.info("In-mem ckpt blocking time is %fs.", time_diff)

    def restore(
        self,
        *,
        step: Optional[int] = None,
        state: Union[Nested[Tensor], Nested[TensorSpec]],
    ) -> Tuple[Optional[int], Nested[Tensor]]:
        """See `BaseCheckpointer.restore` for details."""
        start_t = time.perf_counter()
        cfg: OrbaxEmergencyCheckpointer.Config = self.config
        state_with_tensors = jax.tree.map(
            lambda x: x if isinstance(x, (Tensor, TensorSpec)) else None, state
        )
        tensor_manager = self._get_tensor_manager(state_with_tensors)
        if step is None:
            # Find the intersection of the checkpoint steps managed by tensor and non-tensor
            # manager, and then use the latest step in the intersection for restore. `all_steps`
            # from tensor manager contains both local and persistent checkpoints.
            common_steps = set(tensor_manager.all_steps()).intersection(
                set(
                    (
                        parse_step_from_dir(d)
                        for d in self._non_tensor_manager.checkpoint_paths(
                            self._non_tensor_manager.config.dir
                        )
                    )
                )
            )
            if not common_steps:
                logging.warning("Could not find any completed checkpoints under %s.", cfg.dir)
                return None, state
            step = max(common_steps)

        restore_step, state = self._non_tensor_manager.restore(step=step, state=state)
        assert step == restore_step

        restored_state_with_tensors = tensor_manager.restore(
            step=step,
            args=ocp.args.PyTreeRestore(item=self._get_abstract_state(state_with_tensors)),
        )
        # Merge non-tensor and tensor states by replacing leaves of the non-tensor Pytree with the
        # not-None leaves of the tensor Pytree.
        restored_state = jax.tree.map(
            lambda non_tensor, tensor: non_tensor if tensor is None else tensor,
            state,
            restored_state_with_tensors,
        )
        time_diff = time.perf_counter() - start_t
        logging.info("Took %ss to restore emergency checkpoint from %s.", time_diff, cfg.dir)
        return step, restored_state

    def wait_until_finished(self):
        """See `BaseCheckpointer.wait_until_finished` docstring for details."""
        self._non_tensor_manager.wait_until_finished()
        self._tensor_manager.wait_until_finished()

    def stop(self):
        """See `BaseCheckpointer.stop` for details."""
        self._non_tensor_manager.stop()
        self._tensor_manager.close()

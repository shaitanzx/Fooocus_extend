from modules.model_family import ModelFamily
import modules.config
import os






def get_family(checkpoint_filename: str) -> ModelFamily:
    """Detect the `ModelFamily` of a checkpoint by filename.

    Resolves `checkpoint_filename` the same way the pipeline resolves it
    for loading (`modules.fast_checkpoint.resolve_checkpoint_path`), then
    reads only the safetensors header. Results are cached per path with an
    `(mtime, size)` fingerprint; a checkpoint that changes on disk replaces
    its own cache entry rather than serving a stale family or accumulating
    superseded entries.

    Never raises: a checkpoint that cannot be found or whose header cannot
    be parsed resolves to `ModelFamily.UNKNOWN`, so this can be called
    unconditionally from a UI change handler.
    """

    resolved_path = os.path.join(modules.config.paths_checkpoints[0], checkpoint_filename)



    # resolved_path = resolve_checkpoint_path(
    #     checkpoint_filename, modules.config.paths_checkpoints, modules.config.path_fast_checkpoints
    # )

    try:
        file_stat = os.stat(resolved_path)
    except OSError as e:
        logger.warning(f"Cannot stat checkpoint '{checkpoint_filename}' for family detection: {e}")
        return ModelFamily.UNKNOWN

    fingerprint = (file_stat.st_mtime, file_stat.st_size)
    cached = _family_cache.get(resolved_path)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]

    try:
        keys = _read_state_dict_keys(resolved_path)
        family = _detect_family_from_keys(keys)
    except CorruptCheckpointError as e:
        logger.warning(f"Could not detect model family for '{checkpoint_filename}': {e}")
        family = ModelFamily.UNKNOWN

    _family_cache[resolved_path] = (fingerprint, family)
    return family
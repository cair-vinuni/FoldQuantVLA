# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""A frame reader for LeRobot v3 datasets, without the ``lerobot`` package.

Evo-1 pins torch 2.5.1 and transformers 4.39 and builds flash-attn against
them; every ``lerobot`` release that still supports this environment's Python
drags a newer torch in with it, and upstream is explicit that the pinned
attention path is what its numbers were measured on. So the calibration reads
the dataset directly instead: the v3 layout is a parquet index plus parquet or
video frames, and pyarrow and PIL are already in this environment.

This is a *reader*, not a transform — it returns the frame as stored. Every
transform an observation passes through on its way to the model is still
upstream's (``Evo1_server.decode_image_from_list`` resizes and converts, the
``Normalizer`` normalises), and a video-backed dataset is decoded by upstream's
own ``_VideoDecoderLRU`` rather than by anything written here.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodeEntry:
    """One episode's index row: where its frames live and how many there are."""

    episode_index: int
    length: int
    chunk_index: int
    file_index: int


class LeRobotFrames:
    """Random access to ``(episode, step)`` frames of a local LeRobot v3 dataset."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"{self.root} is not a LeRobot dataset (no meta/info.json)")
        self.info: dict[str, Any] = json.loads(info_path.read_text())
        self.features: dict[str, Any] = self.info.get("features", {})
        self._episodes = self._read_episode_index()
        self._decoder: Any = None

    # ---------------------------------------------------------------- index

    def _read_episode_index(self) -> dict[int, EpisodeEntry]:
        files = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"{self.root}/meta/episodes holds no parquet index")
        entries: dict[int, EpisodeEntry] = {}
        for path in files:
            table = pq.read_table(path)
            columns = set(table.column_names)
            required = {"episode_index", "length", "data/chunk_index", "data/file_index"}
            missing = required - columns
            if missing:
                raise ValueError(f"{path} lacks {sorted(missing)}; is this a v3 LeRobot dataset?")
            for ep, length, chunk, file_index in zip(
                table.column("episode_index").to_pylist(),
                table.column("length").to_pylist(),
                table.column("data/chunk_index").to_pylist(),
                table.column("data/file_index").to_pylist(),
                strict=True,
            ):
                entries[int(ep)] = EpisodeEntry(int(ep), int(length), int(chunk), int(file_index))
        return entries

    def episodes(self) -> list[EpisodeEntry]:
        """Every episode the index knows, by ``episode_index``."""
        return [self._episodes[k] for k in sorted(self._episodes)]

    @lru_cache(maxsize=1)  # noqa: B019 - one scan per dataset
    def _episode_files(self) -> dict[int, Path]:
        """``episode_index -> the data file that really holds its frames``.

        Built by reading the episode column (one column, not the frames) of
        every data file present, rather than trusting the index's
        ``data/file_index``: a partially fetched release leaves files that hold
        fewer episodes than the metadata assigns to them, and the metadata is
        not rewritten to match.
        """
        found: dict[int, Path] = {}
        template = self.info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
        data_root = self.root / template.split("/", 1)[0]
        for path in sorted(data_root.rglob("*.parquet")):
            column = pq.read_table(path, columns=["episode_index"]).column("episode_index").to_numpy()
            for value in np.unique(column):
                found.setdefault(int(value), path)
        return found

    def available_episodes(self) -> list[int]:
        """Episodes whose frames are actually readable."""
        return sorted(set(self._episode_files()) & set(self._episodes))

    def _data_path(self, entry: EpisodeEntry) -> Path:
        template = self.info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
        return self.root / template.format(chunk_index=entry.chunk_index, file_index=entry.file_index)

    # ---------------------------------------------------------------- frames

    @lru_cache(maxsize=4)  # noqa: B019 - bounded by the number of data files a run touches
    def _frame_table(self, path: str):
        return pq.read_table(path)

    @lru_cache(maxsize=1)  # noqa: B019 - one small table per dataset
    def _task_map(self) -> dict[int, str]:
        """``task_index -> task text``.

        The v3 ``tasks.parquet`` carries the text as the frame's index, which
        arrow surfaces as ``__index_level_0__``, so the text column is "the one
        that is not the index", not a column called ``task``.
        """
        tasks_path = self.root / "meta" / "tasks.parquet"
        if not tasks_path.is_file():
            return {}
        table = pq.read_table(tasks_path)
        names = table.column_names
        text_column = next((n for n in names if n != "task_index"), None)
        if text_column is None or "task_index" not in names:
            return {}
        return {
            int(i): str(text)
            for i, text in zip(
                table.column("task_index").to_pylist(), table.column(text_column).to_pylist(), strict=True
            )
        }

    def _task_of(self, task_index: int) -> str:
        return self._task_map().get(int(task_index), "")

    def _image(self, value: Any, key: str, timestamp: float) -> np.ndarray:
        """One camera's frame as a uint8 HWC array, however the dataset stores it."""
        dtype = self.features.get(key, {}).get("dtype")
        if dtype == "video":
            return self._video_frame(key, timestamp)
        if isinstance(value, dict) and "bytes" in value:  # HF Image struct
            return np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("RGB"), dtype=np.uint8)
        array = np.asarray(value)
        if array.dtype != np.uint8:
            scaled = array.max() <= 1.0
            array = np.rint(array * 255.0).clip(0, 255).astype(np.uint8) if scaled else array.astype(np.uint8)
        if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        return array

    def _video_frame(self, key: str, timestamp: float) -> np.ndarray:
        """Decode through upstream's own decoder, so a video dataset is read as upstream reads it."""
        if self._decoder is None:
            from dataset.lerobot_dataset_pretrain_mp import _VideoDecoderLRU

            self._decoder = _VideoDecoderLRU(backend="pyav")
        template = self.info.get("video_path")
        if not template:
            raise ValueError(f"{key} is a video feature but meta/info.json has no video_path template")
        raise NotImplementedError(
            "video-backed LeRobot datasets need the per-episode video file mapping; calibrate from a "
            "dataset whose camera features are stored in the parquet (dtype 'image')"
        )

    def frame(self, episode: int, step: int) -> dict[str, Any]:
        """The stored frame at ``(episode, step)``: camera arrays, state, task text."""
        entry = self._episodes.get(int(episode))
        if entry is None:
            raise KeyError(f"episode {episode} is not in the dataset index")
        if not 0 <= step < entry.length:
            raise IndexError(f"step {step} outside episode {episode} (length {entry.length})")
        path = self._episode_files().get(int(episode))
        if path is None:
            raise FileNotFoundError(f"no data file present holds episode {episode}")
        table = self._frame_table(str(path))
        episode_column = np.asarray(table.column("episode_index").to_numpy())
        rows = np.flatnonzero(episode_column == int(episode))
        if rows.size == 0:
            raise KeyError(f"episode {episode} has no frames in {path}")
        row = int(rows[step])
        record = {name: table.column(name)[row].as_py() for name in table.column_names}
        timestamp = float(record.get("timestamp", 0.0))
        out: dict[str, Any] = {
            key: self._image(record[key], key, timestamp) for key in record if key.startswith("observation.images.")
        }
        out["observation.state"] = np.asarray(record["observation.state"], dtype=np.float32)
        out["task"] = record.get("task") or self._task_of(record.get("task_index", 0))
        return out


def ensure_local(dataset_path: str) -> Path:
    """A local dataset root: a directory as given, or a hub id snapshotted into the cache.

    Only the metadata and the data files are fetched — the same subset a
    calibration run reads.
    """
    root = Path(dataset_path).expanduser()
    if (root / "meta" / "info.json").is_file():
        return root.resolve()
    from huggingface_hub import snapshot_download

    logger.info("fetching %s from the hub (meta + data)", dataset_path)
    local = snapshot_download(
        repo_id=dataset_path,
        repo_type="dataset",
        allow_patterns=["meta/*", "meta/**/*", "data/*", "data/**/*"],
    )
    return Path(local)

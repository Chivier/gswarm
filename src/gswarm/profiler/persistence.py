"""
Persistent storage module for fault tolerance
"""

import json
import os
import time
import aiofiles
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger
from datetime import datetime
import pickle


class PersistentStorage:
    """Base class for persistent storage backends"""

    async def save_frame(self, session_id: str, frame: Dict[str, Any]):
        """Save a single frame to persistent storage"""
        raise NotImplementedError

    async def load_frames(self, session_id: str) -> List[Dict[str, Any]]:
        """Load all frames for a session"""
        raise NotImplementedError

    async def save_session_state(self, session_id: str, state: Dict[str, Any]):
        """Save session metadata and state"""
        raise NotImplementedError

    async def load_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session metadata and state"""
        raise NotImplementedError

    async def list_sessions(self) -> List[str]:
        """List all available sessions"""
        raise NotImplementedError

    async def delete_session(self, session_id: str):
        """Delete a session and its data"""
        raise NotImplementedError


class FileBasedStorage(PersistentStorage):
    """File-based persistent storage implementation"""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            # Use cache directory by default
            from gswarm.utils.cache import get_cache_dir

            self.base_dir = get_cache_dir() / "profiler"
        else:
            self.base_dir = Path(base_dir)

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a lock for a session"""
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    def _get_session_dir(self, session_id: str) -> Path:
        """Get directory path for a session"""
        session_dir = self.base_dir / session_id
        session_dir.mkdir(exist_ok=True)
        return session_dir

    async def save_frame(self, session_id: str, frame: Dict[str, Any]):
        """Save a single frame to persistent storage"""
        async with self._get_lock(session_id):
            session_dir = self._get_session_dir(session_id)
            frames_file = session_dir / "frames.jsonl"

            try:
                async with aiofiles.open(frames_file, mode="a") as f:
                    await f.write(json.dumps(frame) + "\n")
            except Exception as e:
                logger.error(f"Failed to save frame for session {session_id}: {e}")
                raise

    async def load_frames(self, session_id: str) -> List[Dict[str, Any]]:
        """Load all frames for a session"""
        async with self._get_lock(session_id):
            session_dir = self._get_session_dir(session_id)
            frames_file = session_dir / "frames.jsonl"

            if not frames_file.exists():
                return []

            frames = []
            try:
                async with aiofiles.open(frames_file, mode="r") as f:
                    async for line in f:
                        if line.strip():
                            frames.append(json.loads(line))
            except Exception as e:
                logger.error(f"Failed to load frames for session {session_id}: {e}")
                raise

            return frames

    async def save_session_state(self, session_id: str, state: Dict[str, Any]):
        """Save session metadata and state"""
        async with self._get_lock(session_id):
            session_dir = self._get_session_dir(session_id)
            state_file = session_dir / "state.json"

            try:
                # Add timestamp
                state["last_updated"] = datetime.now().isoformat()

                # Write to temporary file first for atomicity
                temp_file = state_file.with_suffix(".tmp")
                async with aiofiles.open(temp_file, mode="w") as f:
                    await f.write(json.dumps(state, indent=2))

                # Atomic rename
                temp_file.replace(state_file)

                # Also save checkpoint if requested
                if state.get("create_checkpoint", False):
                    checkpoint_file = session_dir / f"checkpoint_{int(time.time())}.json"
                    async with aiofiles.open(checkpoint_file, mode="w") as f:
                        await f.write(json.dumps(state, indent=2))

            except Exception as e:
                logger.error(f"Failed to save session state for {session_id}: {e}")
                raise

    async def load_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session metadata and state"""
        async with self._get_lock(session_id):
            session_dir = self._get_session_dir(session_id)
            state_file = session_dir / "state.json"

            if not state_file.exists():
                return None

            try:
                async with aiofiles.open(state_file, mode="r") as f:
                    content = await f.read()
                    return json.loads(content)
            except Exception as e:
                logger.error(f"Failed to load session state for {session_id}: {e}")
                # Try to load from latest checkpoint
                checkpoint_files = sorted(session_dir.glob("checkpoint_*.json"))
                if checkpoint_files:
                    try:
                        async with aiofiles.open(checkpoint_files[-1], mode="r") as f:
                            content = await f.read()
                            return json.loads(content)
                    except Exception:
                        pass
                return None

    async def list_sessions(self) -> List[str]:
        """List all available sessions"""
        sessions = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and (item / "state.json").exists():
                sessions.append(item.name)
        return sessions

    async def delete_session(self, session_id: str):
        """Delete a session and its data"""
        async with self._get_lock(session_id):
            session_dir = self._get_session_dir(session_id)
            if session_dir.exists():
                import shutil

                shutil.rmtree(session_dir)

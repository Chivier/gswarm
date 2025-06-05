"""
Session management module for handling multiple concurrent profiling sessions
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger
from gswarm.profiler.persistence import PersistentStorage, FileBasedStorage


class ProfilingSession:
    """Represents a single profiling session"""

    def __init__(self, session_id: str, name: str, storage: PersistentStorage):
        self.session_id = session_id
        self.name = name
        self.storage = storage
        self.start_time = datetime.now()
        self.end_time = None
        self.is_active = True
        self.frame_count = 0
        self.connected_clients = set()
        self.output_filename = f"{name}.json" if name else f"session_{session_id}.json"

        # Statistics accumulators
        self.gpu_total_util: Dict[str, float] = {}
        self.gpu_util_count: Dict[str, int] = {}
        self.gpu_total_memory: Dict[str, float] = {}
        self.gpu_memory_count: Dict[str, int] = {}

    async def add_frame(self, frame: Dict[str, Any]):
        """Add a frame to this session"""
        self.frame_count += 1
        frame["session_id"] = self.session_id
        frame["session_name"] = self.name

        # Update statistics
        for i, gpu_id in enumerate(frame.get("gpu_id", [])):
            util = float(frame["gpu_util"][i])
            memory = float(frame["gpu_memory"][i])

            self.gpu_total_util[gpu_id] = self.gpu_total_util.get(gpu_id, 0.0) + util
            self.gpu_util_count[gpu_id] = self.gpu_util_count.get(gpu_id, 0) + 1
            self.gpu_total_memory[gpu_id] = self.gpu_total_memory.get(gpu_id, 0.0) + memory
            self.gpu_memory_count[gpu_id] = self.gpu_memory_count.get(gpu_id, 0) + 1

        # Persist frame immediately
        await self.storage.save_frame(self.session_id, frame)

    async def stop(self):
        """Stop this session"""
        self.is_active = False
        self.end_time = datetime.now()
        await self.save_state()

    async def save_state(self):
        """Save session state to persistent storage"""
        state = {
            "session_id": self.session_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "is_active": self.is_active,
            "frame_count": self.frame_count,
            "output_filename": self.output_filename,
            "connected_clients": list(self.connected_clients),
            "gpu_stats": {
                "total_util": self.gpu_total_util,
                "util_count": self.gpu_util_count,
                "total_memory": self.gpu_total_memory,
                "memory_count": self.gpu_memory_count,
            },
        }
        await self.storage.save_session_state(self.session_id, state)

    async def recover(self, state: Dict[str, Any]):
        """Recover session from saved state"""
        self.start_time = datetime.fromisoformat(state["start_time"])
        if state.get("end_time"):
            self.end_time = datetime.fromisoformat(state["end_time"])
        self.is_active = state.get("is_active", False)
        self.frame_count = state.get("frame_count", 0)
        self.output_filename = state.get("output_filename", self.output_filename)
        self.connected_clients = set(state.get("connected_clients", []))

        # Recover statistics
        gpu_stats = state.get("gpu_stats", {})
        self.gpu_total_util = gpu_stats.get("total_util", {})
        self.gpu_util_count = gpu_stats.get("util_count", {})
        self.gpu_total_memory = gpu_stats.get("total_memory", {})
        self.gpu_memory_count = gpu_stats.get("memory_count", {})

    def get_summary(self) -> Dict[str, Any]:
        """Get session summary statistics"""
        summary_by_device = {}

        for gpu_id, total_util in self.gpu_total_util.items():
            count = self.gpu_util_count.get(gpu_id, 0)
            if count > 0:
                avg_util = total_util / count
                summary_by_device.setdefault(gpu_id, {})["avg_gpu_util"] = f"{avg_util:.2f}"

        for gpu_id, total_mem in self.gpu_total_memory.items():
            count = self.gpu_memory_count.get(gpu_id, 0)
            if count > 0:
                avg_mem = total_mem / count
                summary_by_device.setdefault(gpu_id, {})["avg_gpu_memory"] = f"{avg_mem:.2f}"

        return summary_by_device


class SessionManager:
    """Manages multiple concurrent profiling sessions"""

    def __init__(self, storage: Optional[PersistentStorage] = None):
        self.storage = storage or FileBasedStorage()
        self.sessions: Dict[str, ProfilingSession] = {}
        self._lock = asyncio.Lock()
        self._checkpoint_interval = 30  # seconds
        self._checkpoint_task = None

    async def initialize(self):
        """Initialize session manager and recover any existing sessions"""
        # Recover existing sessions from storage
        session_ids = await self.storage.list_sessions()
        for session_id in session_ids:
            state = await self.storage.load_session_state(session_id)
            if state and state.get("is_active", False):
                logger.info(f"Recovering active session: {state['name']} ({session_id})")
                session = ProfilingSession(session_id, state["name"], self.storage)
                await session.recover(state)
                # Mark as recovered but inactive until explicitly resumed
                session.is_active = False
                self.sessions[session_id] = session

        # Start checkpoint task
        self._checkpoint_task = asyncio.create_task(self._periodic_checkpoint())

    async def create_session(self, name: Optional[str] = None) -> ProfilingSession:
        """Create a new profiling session"""
        async with self._lock:
            session_id = str(uuid.uuid4())
            if not name:
                name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            session = ProfilingSession(session_id, name, self.storage)
            self.sessions[session_id] = session
            await session.save_state()

            logger.info(f"Created new session: {name} ({session_id})")
            return session

    async def get_session(self, session_id: str) -> Optional[ProfilingSession]:
        """Get a session by ID"""
        return self.sessions.get(session_id)

    async def get_session_by_name(self, name: str) -> Optional[ProfilingSession]:
        """Get a session by name"""
        for session in self.sessions.values():
            if session.name == name:
                return session
        return None

    async def list_sessions(self, active_only: bool = False) -> List[ProfilingSession]:
        """List all sessions"""
        sessions = list(self.sessions.values())
        if active_only:
            sessions = [s for s in sessions if s.is_active]
        return sessions

    async def stop_session(self, session_id: str):
        """Stop a specific session"""
        async with self._lock:
            session = self.sessions.get(session_id)
            if session and session.is_active:
                await session.stop()
                logger.info(f"Stopped session: {session.name} ({session_id})")

    async def stop_all_sessions(self):
        """Stop all active sessions"""
        async with self._lock:
            for session in self.sessions.values():
                if session.is_active:
                    await session.stop()

    async def resume_session(self, session_id: str) -> Optional[ProfilingSession]:
        """Resume a stopped session"""
        async with self._lock:
            session = self.sessions.get(session_id)
            if session and not session.is_active:
                session.is_active = True
                session.end_time = None
                await session.save_state()
                logger.info(f"Resumed session: {session.name} ({session_id})")
                return session
        return None

    async def export_session(self, session_id: str) -> Dict[str, Any]:
        """Export session data to final format"""
        session = self.sessions.get(session_id)
        if not session:
            return None

        frames = await self.storage.load_frames(session_id)
        summary = session.get_summary()

        return {
            "session_name": session.name,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "frames": frames,
            "summary_by_device": summary,
        }

    async def _periodic_checkpoint(self):
        """Periodically checkpoint all active sessions"""
        while True:
            try:
                await asyncio.sleep(self._checkpoint_interval)
                async with self._lock:
                    for session in self.sessions.values():
                        if session.is_active:
                            await session.save_state()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic checkpoint: {e}")

    async def shutdown(self):
        """Shutdown session manager"""
        if self._checkpoint_task:
            self._checkpoint_task.cancel()
            try:
                await self._checkpoint_task
            except asyncio.CancelledError:
                pass

        # Save all session states
        async with self._lock:
            for session in self.sessions.values():
                await session.save_state()

"""Data pool manager implementation"""

import os
import hashlib
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import uuid
from loguru import logger
import aiofiles
import asyncio


@dataclass
class DataLocation:
    device: str
    path: str
    status: str = "available"  # available, moving, deleted


@dataclass
class DataChunk:
    chunk_id: str
    chunk_type: str  # input, output, intermediate
    size: int
    locations: List[DataLocation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)  # Models/services using this


class DataPoolManager:
    """Manages distributed data chunks across devices"""

    def __init__(self, base_path: str = "~/.gswarm/data_pool"):
        self.base_path = Path(base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.chunks: Dict[str, DataChunk] = {}
        self._load_metadata()

    def _load_metadata(self):
        """Load chunk metadata from disk"""
        metadata_file = self.base_path / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                    for chunk_data in data.get("chunks", []):
                        chunk = DataChunk(**chunk_data)
                        self.chunks[chunk.chunk_id] = chunk
                logger.info(f"Loaded {len(self.chunks)} data chunks from metadata")
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")

    def _save_metadata(self):
        """Save chunk metadata to disk"""
        metadata_file = self.base_path / "metadata.json"
        try:
            data = {
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "chunk_type": c.chunk_type,
                        "size": c.size,
                        "locations": [
                            {"device": loc.device, "path": loc.path, "status": loc.status} for loc in c.locations
                        ],
                        "metadata": c.metadata,
                        "references": c.references,
                    }
                    for c in self.chunks.values()
                ]
            }
            with open(metadata_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    async def create_chunk(
        self,
        data: bytes,
        chunk_type: str = "input",
        device: str = "dram",
        format: str = "tensor",
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Create a new data chunk"""
        chunk_id = f"chunk-{uuid.uuid4().hex[:12]}"

        # Calculate checksum
        checksum = hashlib.sha256(data).hexdigest()

        # Check for duplicates
        for chunk in self.chunks.values():
            if chunk.metadata.get("checksum") == checksum:
                logger.info(f"Data already exists as chunk {chunk.chunk_id}")
                return chunk.chunk_id

        # Create storage path
        chunk_path = self._get_chunk_path(chunk_id, device)
        chunk_path.parent.mkdir(parents=True, exist_ok=True)

        # Write data
        async with aiofiles.open(chunk_path, "wb") as f:
            await f.write(data)

        # Create chunk metadata
        chunk = DataChunk(
            chunk_id=chunk_id,
            chunk_type=chunk_type,
            size=len(data),
            locations=[DataLocation(device=device, path=str(chunk_path))],
            metadata={
                "format": format,
                "checksum": checksum,
                "created_at": datetime.now().isoformat(),
                "created_by": "user",
                "access_count": 0,
                **(metadata or {}),
            },
        )

        self.chunks[chunk_id] = chunk
        self._save_metadata()

        logger.info(f"Created data chunk {chunk_id} on {device} ({len(data) / 1e6:.2f} MB)")
        return chunk_id

    async def get_chunk_data(self, chunk_id: str, device: Optional[str] = None) -> bytes:
        """Read chunk data from storage"""
        if chunk_id not in self.chunks:
            raise ValueError(f"Chunk {chunk_id} not found")

        chunk = self.chunks[chunk_id]

        # Find available location
        location = None
        if device:
            location = next(
                (loc for loc in chunk.locations if loc.device == device and loc.status == "available"), None
            )
        else:
            location = next((loc for loc in chunk.locations if loc.status == "available"), None)

        if not location:
            raise ValueError(f"No available location for chunk {chunk_id}")

        # Read data
        async with aiofiles.open(location.path, "rb") as f:
            data = await f.read()

        # Update access metadata
        chunk.metadata["last_accessed"] = datetime.now().isoformat()
        chunk.metadata["access_count"] = chunk.metadata.get("access_count", 0) + 1
        self._save_metadata()

        return data

    async def move_chunk(
        self,
        chunk_id: str,
        target_device: str,
        source_device: Optional[str] = None,
        keep_source: bool = True,
    ):
        """Move chunk between devices"""
        if chunk_id not in self.chunks:
            raise ValueError(f"Chunk {chunk_id} not found")

        chunk = self.chunks[chunk_id]

        # Find source location
        if source_device:
            source_loc = next((loc for loc in chunk.locations if loc.device == source_device), None)
        else:
            source_loc = next((loc for loc in chunk.locations if loc.status == "available"), None)

        if not source_loc:
            raise ValueError(f"No source location found for chunk {chunk_id}")

        # Check if already on target
        if any(loc.device == target_device for loc in chunk.locations):
            logger.info(f"Chunk {chunk_id} already on {target_device}")
            return

        # Read data
        async with aiofiles.open(source_loc.path, "rb") as f:
            data = await f.read()

        # Write to target
        target_path = self._get_chunk_path(chunk_id, target_device)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(target_path, "wb") as f:
            await f.write(data)

        # Update locations
        chunk.locations.append(DataLocation(device=target_device, path=str(target_path), status="available"))

        # Remove source if requested
        if not keep_source:
            os.unlink(source_loc.path)
            chunk.locations.remove(source_loc)

        self._save_metadata()
        logger.info(f"Moved chunk {chunk_id} from {source_loc.device} to {target_device}")

    async def delete_chunk(self, chunk_id: str, force: bool = False):
        """Delete a data chunk"""
        if chunk_id not in self.chunks:
            raise ValueError(f"Chunk {chunk_id} not found")

        chunk = self.chunks[chunk_id]

        # Check references
        if chunk.references and not force:
            raise ValueError(f"Chunk {chunk_id} is referenced by: {', '.join(chunk.references)}")

        # Delete all locations
        for location in chunk.locations:
            try:
                if os.path.exists(location.path):
                    os.unlink(location.path)
            except Exception as e:
                logger.error(f"Failed to delete {location.path}: {e}")

        # Remove from registry
        del self.chunks[chunk_id]
        self._save_metadata()

        logger.info(f"Deleted chunk {chunk_id}")

    def add_reference(self, chunk_id: str, reference: str):
        """Add a reference to a chunk"""
        if chunk_id in self.chunks:
            chunk = self.chunks[chunk_id]
            if reference not in chunk.references:
                chunk.references.append(reference)
                self._save_metadata()

    def remove_reference(self, chunk_id: str, reference: str):
        """Remove a reference from a chunk"""
        if chunk_id in self.chunks:
            chunk = self.chunks[chunk_id]
            if reference in chunk.references:
                chunk.references.remove(reference)
                self._save_metadata()

    def list_chunks(
        self,
        device: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List chunks with optional filters"""
        chunks = []

        for chunk in self.chunks.values():
            # Apply filters
            if device and not any(loc.device == device for loc in chunk.locations):
                continue
            if chunk_type and chunk.chunk_type != chunk_type:
                continue

            chunks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": chunk.chunk_type,
                    "size": chunk.size,
                    "format": chunk.metadata.get("format", "unknown"),
                    "locations": [{"device": loc.device, "status": loc.status} for loc in chunk.locations],
                    "metadata": chunk.metadata,
                    "references": chunk.references,
                }
            )

        return chunks

    def get_chunk_info(self, chunk_id: str) -> Dict[str, Any]:
        """Get detailed chunk information"""
        if chunk_id not in self.chunks:
            raise ValueError(f"Chunk {chunk_id} not found")

        chunk = self.chunks[chunk_id]
        return {
            "chunk_id": chunk.chunk_id,
            "chunk_type": chunk.chunk_type,
            "size": chunk.size,
            "locations": [{"device": loc.device, "path": loc.path, "status": loc.status} for loc in chunk.locations],
            "metadata": chunk.metadata,
            "references": chunk.references,
        }

    def _get_chunk_path(self, chunk_id: str, device: str) -> Path:
        """Get storage path for a chunk on a device"""
        # Map device to storage location
        if device == "disk":
            base = self.base_path / "disk"
        elif device == "dram":
            base = Path("/dev/shm") / "gswarm" / "data_pool"
        elif device.startswith("gpu"):
            # GPU memory would need special handling
            base = self.base_path / device
        else:
            base = self.base_path / device

        return base / chunk_id[:2] / chunk_id


# Global instance
data_pool = DataPoolManager()

"""
Resilient head node implementation with session management and persistence
"""
# ... existing code ...
# Add these modifications to head.py:

# At the top, add imports:
from .session_manager import SessionManager, ProfilingSession
from .persistence import FileBasedStorage

# Modify HeadNodeState class:
class HeadNodeState:
    def __init__(self):
        # ... existing attributes ...
        
        # Session management
        self.session_manager = SessionManager()
        self.active_sessions: Dict[str, ProfilingSession] = {}
        
        # Health monitoring
        self.client_last_seen: Dict[str, float] = {}
        self.client_health_timeout = 30  # seconds
        
    async def initialize(self):
        """Initialize head node state"""
        await self.session_manager.initialize()
        
    async def mark_client_active(self, client_id: str):
        """Mark a client as active"""
        self.client_last_seen[client_id] = time.time()
        
    async def check_client_health(self):
        """Check health of connected clients"""
        current_time = time.time()
        disconnected_clients = []
        
        for client_id, last_seen in self.client_last_seen.items():
            if current_time - last_seen > self.client_health_timeout:
                disconnected_clients.append(client_id)
                
        for client_id in disconnected_clients:
            logger.warning(f"Client {client_id} appears to be disconnected (no data for {self.client_health_timeout}s)")
            # Mark data as potentially incomplete
            for session in self.active_sessions.values():
                if client_id in session.connected_clients:
                    session.connected_clients.remove(client_id)
                    
        return disconnected_clients

# Add health monitoring task:
async def health_monitor_task():
    """Periodically check client health"""
    while True:
        try:
            await asyncio.sleep(10)
            disconnected = await state.check_client_health()
            if disconnected:
                logger.info(f"Health check found {len(disconnected)} disconnected clients")
        except Exception as e:
            logger.error(f"Error in health monitor: {e}")

# Modify the profiling collection to support multiple sessions:
async def collect_and_store_frame():
    """Periodically collects data from clients for all active sessions"""
    while True:
        await asyncio.sleep(1)
        
        async with state.data_lock:
            active_sessions = [s for s in state.active_sessions.values() if s.is_active]
            
            if not active_sessions:
                continue
                
            # Collect current frame data
            current_time = datetime.datetime.now().isoformat()
            active_clients_data = {k: v for k, v in state.latest_client_data.items() 
                                 if k in state.connected_clients}
            
            # Process data for each active session
            for session in active_sessions:
                frame = {
                    "frame_id": session.frame_count + 1,
                    "time": current_time,
                    "gpu_id": [],
                    "gpu_util": [],
                    "gpu_memory": [],
                }
                
                if state.enable_bandwidth_profiling:
                    frame["dram_bandwidth"] = []
                    frame["dram_bandwidth_rx"] = []
                    frame["dram_bandwidth_tx"] = []
                    
                if state.enable_nvlink_profiling:
                    frame["gpu_bandwidth"] = []
                
                # Collect metrics from all clients
                for client_id, client_payload in active_clients_data.items():
                    # ... existing frame collection logic ...
                    
                # Add frame to session
                try:
                    await session.add_frame(frame)
                except Exception as e:
                    logger.error(f"Failed to add frame to session {session.name}: {e}") 
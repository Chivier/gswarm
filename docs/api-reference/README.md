# API Reference

Complete API documentation for GSwarm components.

## Documents

### Host APIs
- [**Host API Reference**](Host-API-Reference.md) - Complete REST and gRPC API documentation for the host node
  - Profiler HTTP and gRPC APIs
  - Model Host and Client APIs
  - Request/response formats
  - Error handling
  - Authentication

### Client APIs
- [**Client API Reference**](Client-API-Reference.md) - Client management API documentation
  - Client management endpoints
  - Status monitoring
  - Configuration APIs
  - Error responses

## API Conventions

All GSwarm APIs follow consistent patterns:

- **REST APIs**: Use JSON for request/response with standard HTTP methods
- **gRPC APIs**: Use Protocol Buffers for efficient binary communication
- **Error Handling**: Consistent error response formats across all endpoints
- **Authentication**: Token-based authentication where applicable

## Quick Reference

### Base URLs
- **Profiler gRPC**: `grpc://host:8090`
- **Profiler HTTP**: `http://host:8091`
- **Model Host API**: `http://host:9010`
- **Client API**: `http://client:10000`

### Common Response Format
```json
{
    "success": true,
    "data": { ... },
    "error": null
}
```

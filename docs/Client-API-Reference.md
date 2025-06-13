# gswarm Client API Reference

The gswarm client provides a REST API for management and some communication functions. This document offers a comprehensive reference for all available APIs.

## Base URLs

> If the default port is unavailable, the client will search for an available port between 10000 and 20000, checking one by one.

-  **Client**: `http://client:10000`

## REST API Conventions

### Request Format
-  Content-Type: `application/json`
-  Accept: `application/json`

### Response Format
```json
{
    "success": true,
    "message": "...",
    "data": { ... },
    "error": null
}
```

## Client Management APIs

#### Shutdown Client
```http
GET /api/v1/shutdown
```

Response:
```json
{
    "message": "Client shutdown initiated"
}
```

> Not Implemented 
### Peer Management

#### Update Peer List
```http
POST /api/v1/peer/update
```

#### Remove Peer
```http
DELETE /api/v1/peer/remove
```

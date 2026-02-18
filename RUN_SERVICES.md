# Starting the infra services

## Configuration

Before starting services, configure your `.env` file (copy from `.example.env` if needed):

1. **Set container creation flags** for each service:
   - `CREATE_POSTGRES_CONTAINER=true` - Create Postgres container (default: uses external)
   - `CREATE_REDIS_CONTAINER=true` - Create Redis container (default: uses external)
   - `CREATE_MINIO_CONTAINER=true` - Create MinIO container (default: `true`)
   - `CREATE_CLICKHOUSE_CONTAINER=true` - Create ClickHouse container (default: `true`)

2. **If using external services** (flag set to `false`), configure:
   - External service hostnames, ports, and credentials in `.env`
   - See `.example.env` for all available configuration options

## Starting Services

### Option 1: Start all infrastructure services (if CREATE_*_CONTAINER=true)

```bash
# Start Postgres (if CREATE_POSTGRES_CONTAINER=true)
docker compose -f postgres-compose.yml up -d

# Start Redis (if CREATE_REDIS_CONTAINER=true)
docker compose -f redis-compose.yml up -d

# Start MinIO (if CREATE_MINIO_CONTAINER=true)
docker compose -f minio-compose.yml up -d

# Start ClickHouse (if CREATE_CLICKHOUSE_CONTAINER=true)
docker compose -f clickhouse-compose.yml up -d

# Start Langfuse (main app)
docker compose -f docker-compose.yml up -d
```

### Option 2: Start only services you need

If you're using external services for some components, only start the containers you need:

```bash
# Example: Using external Postgres and Redis, but containers for MinIO and ClickHouse
docker compose -f minio-compose.yml up -d
docker compose -f clickhouse-compose.yml up -d
docker compose -f docker-compose.yml up -d
```

### Quick Start (all services)

```bash
# Start all infrastructure services at once
docker compose -f postgres-compose.yml -f redis-compose.yml -f minio-compose.yml -f clickhouse-compose.yml up -d

# Then start the main app
docker compose -f docker-compose.yml up -d
```

---

## Stopping services

```bash
# Stop main app
docker compose -f docker-compose.yml down

# Stop infrastructure services (only if you created containers)
docker compose -f clickhouse-compose.yml down
docker compose -f minio-compose.yml down
docker compose -f redis-compose.yml down
docker compose -f postgres-compose.yml down
```

### Stop all at once

```bash
docker compose -f docker-compose.yml -f clickhouse-compose.yml -f minio-compose.yml -f redis-compose.yml -f postgres-compose.yml down
```

---

## Viewing logs

```bash
# Follow logs for all services in main compose
docker compose -f docker-compose.yml logs -f

# Follow logs for a specific service
docker compose -f docker-compose.yml logs -f langfuse-web
docker compose -f docker-compose.yml logs -f langfuse-worker

# Infrastructure service logs
docker compose -f postgres-compose.yml logs -f postgres
docker compose -f redis-compose.yml logs -f redis
docker compose -f minio-compose.yml logs -f minio
docker compose -f clickhouse-compose.yml logs -f clickhouse
```

---

## Checking status

```bash
# See running containers and health
docker compose -f docker-compose.yml ps
docker compose -f postgres-compose.yml ps
docker compose -f redis-compose.yml ps
docker compose -f minio-compose.yml ps
docker compose -f clickhouse-compose.yml ps
```

---

## Restarting a single service

```bash
docker compose -f docker-compose.yml restart langfuse-web
docker compose -f docker-compose.yml restart langfuse-worker
```

---

## Pulling latest images & redeploying

```bash
# Pull and restart infrastructure services
docker compose -f postgres-compose.yml pull && docker compose -f postgres-compose.yml up -d
docker compose -f redis-compose.yml pull && docker compose -f redis-compose.yml up -d
docker compose -f minio-compose.yml pull && docker compose -f minio-compose.yml up -d
docker compose -f clickhouse-compose.yml pull && docker compose -f clickhouse-compose.yml up -d

# Pull and restart main app
docker compose -f docker-compose.yml pull && docker compose -f docker-compose.yml up -d
```

---

## Full teardown (keeps data volumes)

```bash
docker compose -f docker-compose.yml down
docker compose -f clickhouse-compose.yml down
docker compose -f minio-compose.yml down
docker compose -f redis-compose.yml down
docker compose -f postgres-compose.yml down
```

## Full teardown + delete all data volumes

```bash
# ⚠️ Destructive — deletes all stored data
docker compose -f docker-compose.yml down -v
docker compose -f clickhouse-compose.yml down -v
docker compose -f minio-compose.yml down -v
docker compose -f redis-compose.yml down -v
docker compose -f postgres-compose.yml down -v
```

---

## Service Configuration Details

### Using External Services

If you set `CREATE_*_CONTAINER=false` in your `.env` file, you need to provide external service credentials:

**Postgres:**
- `DATABASE_URL` - Full connection string to external Postgres

**Redis:**
- `REDIS_HOST` - External Redis hostname
- `REDIS_PORT` - External Redis port
- `REDIS_AUTH` - Redis password

**MinIO/S3:**
- `MINIO_EXTERNAL_HOST` - External MinIO/S3 hostname
- `MINIO_EXTERNAL_PORT` - External MinIO/S3 port
- `MINIO_EXTERNAL_ACCESS_KEY_ID` - Access key
- `MINIO_EXTERNAL_SECRET_ACCESS_KEY` - Secret key

**ClickHouse:**
- `CLICKHOUSE_EXTERNAL_HOST` - External ClickHouse hostname
- `CLICKHOUSE_EXTERNAL_PORT` - Native port (default: 9000)
- `CLICKHOUSE_EXTERNAL_HTTP_PORT` - HTTP port (default: 8123)
- `CLICKHOUSE_USER` - ClickHouse username
- `CLICKHOUSE_PASSWORD` - ClickHouse password

### Default Ports (when using containers)

- **Postgres**: `127.0.0.1:5434` (container port 5432)
- **Redis**: `127.0.0.1:6379`
- **MinIO API**: `127.0.0.1:9090` (container port 9000)
- **MinIO Console**: `127.0.0.1:9091` (container port 9001)
- **ClickHouse HTTP**: `127.0.0.1:8123`
- **ClickHouse Native**: `127.0.0.1:9000`
- **Langfuse Web**: `127.0.0.1:8501`
- **Langfuse Worker**: `127.0.0.1:3030`

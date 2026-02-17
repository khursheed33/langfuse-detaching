# Starting the infra services

```bash
# Start Postgres
docker compose -f postgres-compose.yml up -d

# Start Redis
docker compose -f redis-compose.yml up -d

# Start Langfuse (main app)
docker compose -f docker-compose.yml up -d
```

---

## Stopping services

```bash
docker compose -f postgres-compose.yml down
docker compose -f redis-compose.yml down
docker compose -f docker-compose.yml down
```

---

## Viewing logs

```bash
# Follow logs for all services in a file
docker compose -f docker-compose.yml logs -f

# Follow logs for a specific service
docker compose -f docker-compose.yml logs -f langfuse-web
docker compose -f postgres-compose.yml logs -f postgres
docker compose -f redis-compose.yml logs -f redis
```

---

## Checking status

```bash
# See running containers and health
docker compose -f docker-compose.yml ps
docker compose -f postgres-compose.yml ps
docker compose -f redis-compose.yml ps
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
docker compose -f postgres-compose.yml pull && docker compose -f postgres-compose.yml up -d
docker compose -f redis-compose.yml pull && docker compose -f redis-compose.yml up -d
docker compose -f docker-compose.yml pull && docker compose -f docker-compose.yml up -d
```

---

## Full teardown (keeps data volumes)

```bash
docker compose -f docker-compose.yml down
docker compose -f redis-compose.yml down
docker compose -f postgres-compose.yml down
```

## Full teardown + delete all data volumes

```bash
# ⚠️ Destructive — deletes all stored data
docker compose -f docker-compose.yml down -v
docker compose -f redis-compose.yml down -v
docker compose -f postgres-compose.yml down -v
```

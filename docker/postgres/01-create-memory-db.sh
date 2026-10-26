#!/bin/sh
# Creates the conversation-memory database next to the business database.
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
  -c "CREATE DATABASE \"${MEMORY_POSTGRES_DB:-agent_memory}\""

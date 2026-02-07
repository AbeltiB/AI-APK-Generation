"""
Database initialization script.

Creates all required tables, indexes, and initial data.
Run this before starting the service for the first time.

Run with:
    poetry run python -m app.scripts.db_init
"""

import asyncio
import asyncpg
from loguru import logger
import sys

from app.config import settings


# ---------------------------------------------------------------------------
# TABLE CREATION
# ---------------------------------------------------------------------------

async def create_tables(conn: asyncpg.Connection) -> None:
    """Create all database tables."""

    logger.info("📦 Creating tables...")

    # Enable required extensions
    await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

    # Conversations table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id VARCHAR(255) NOT NULL,
            session_id VARCHAR(255) NOT NULL,
            messages JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
    """)

    # Projects table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id VARCHAR(255) NOT NULL,
            project_name VARCHAR(255),
            architecture JSONB NOT NULL,
            layout JSONB NOT NULL,
            blockly JSONB NOT NULL,

            -- Project State integration
            state_project_id VARCHAR(255),
            state_version INTEGER DEFAULT 1,
            state_schema_version INTEGER,

            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
    """)

    # User preferences table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id VARCHAR(255) PRIMARY KEY,
            preferences JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
    """)

    # Request metrics table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS request_metrics (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            task_id VARCHAR(255) NOT NULL,
            user_id VARCHAR(255) NOT NULL,
            stage VARCHAR(50) NOT NULL,
            duration_ms INTEGER NOT NULL,
            success BOOLEAN NOT NULL,
            error_message TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
    """)

    logger.info("✅ Tables created successfully")


# ---------------------------------------------------------------------------
# INDEXES
# ---------------------------------------------------------------------------

async def create_indexes(conn: asyncpg.Connection) -> None:
    """Create database indexes for performance."""

    logger.info("⚡ Creating indexes...")

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_user_session
        ON conversations(user_id, session_id);
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_created_at
        ON conversations(created_at DESC);
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_projects_user_id
        ON projects(user_id);
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_projects_updated_at
        ON projects(updated_at DESC);
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_metrics_task_id
        ON request_metrics(task_id);
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_metrics_user_id
        ON request_metrics(user_id);
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_metrics_created_at
        ON request_metrics(created_at DESC);
    """)

    logger.info("✅ Indexes created successfully")


# ---------------------------------------------------------------------------
# SEED DATA (DEV ONLY)
# ---------------------------------------------------------------------------

async def seed_test_data(conn: asyncpg.Connection) -> None:
    """Insert test data for development."""

    logger.info("🌱 Seeding test data...")

    await conn.execute("""
        INSERT INTO user_preferences (user_id, preferences)
        VALUES (
            'test_user_1',
            '{"theme":"dark","component_style":"minimal"}'::jsonb
        )
        ON CONFLICT (user_id) DO NOTHING;
    """)

    logger.info("✅ Test data seeded")


# ---------------------------------------------------------------------------
# VERIFICATION
# ---------------------------------------------------------------------------

async def verify_tables(conn: asyncpg.Connection) -> None:
    """Verify all tables exist and are accessible."""

    logger.info("🔍 Verifying database schema...")

    rows = await conn.fetch("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)

    found = {row["table_name"] for row in rows}
    expected = {
        "conversations",
        "projects",
        "user_preferences",
        "request_metrics",
    }

    if not expected.issubset(found):
        missing = expected - found
        raise RuntimeError(f"Missing tables: {', '.join(missing)}")

    logger.info("✅ All required tables verified")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

async def main() -> None:
    print("\n" + "=" * 70)
    print("🚀 DATABASE INITIALIZATION")
    print("=" * 70)

    try:
        logger.info(
            f"Connecting to PostgreSQL "
            f"{settings.postgres_host}:{settings.postgres_port}/"
            f"{settings.postgres_db}"
        )

        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
        )

        logger.info("✅ Connected to PostgreSQL")

        await create_tables(conn)
        await create_indexes(conn)

        if settings.debug:
            await seed_test_data(conn)

        await verify_tables(conn)

        await conn.close()

        print("\n" + "=" * 70)
        print("✅ DATABASE INITIALIZATION COMPLETE")
        print("=" * 70)
        print("\nStart the service with:")
        print("  poetry run uvicorn app.main:app --reload")
        print("\n" + "=" * 70 + "\n")

    except Exception as e:
        logger.error("❌ Database initialization failed", exc_info=e)

        print("\n" + "=" * 70)
        print("❌ INITIALIZATION FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  • Ensure PostgreSQL is running")
        print("  • Verify .env credentials")
        print("  • Check database exists")
        print("\n" + "=" * 70 + "\n")

        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
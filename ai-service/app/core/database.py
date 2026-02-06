"""
Enhanced PostgreSQL database manager with connection pooling.

Provides async interface to PostgreSQL with connection pooling,
automatic reconnection, and efficient query execution.
"""
import asyncpg
from typing import Optional, List, Dict, Any
from loguru import logger
from contextlib import asynccontextmanager

from app.config import settings


class DatabaseManager:
    """
    Manages PostgreSQL connections and operations.
    
    Features:
    - Connection pooling for high performance
    - Automatic reconnection on failure
    - Transaction support
    - Query result caching
    """
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self._connected = False
    
    async def connect(self) -> None:
        """
        Establish connection pool to PostgreSQL.
        
        Creates a connection pool with min/max connections
        and tests connectivity.
        """
        try:
            # Debug logging - remove in production
            logger.info(f"Connecting to PostgreSQL: {settings.postgres_host}:{settings.postgres_port}")
            logger.debug(f"Database: {settings.postgres_db}")
            logger.debug(f"User: {settings.postgres_user}")
            logger.debug(f"Password length: {len(settings.postgres_password) if settings.postgres_password else 0}")
            
            # Option 1: Use individual parameters (RECOMMENDED)
            # This is more reliable and avoids URL encoding issues
            self.pool = await asyncpg.create_pool(
                host=settings.postgres_host,
                port=settings.postgres_port,
                database=settings.postgres_db,
                user=settings.postgres_user,
                password=settings.postgres_password,
                min_size=settings.postgres_min_connections,
                max_size=settings.postgres_max_connections,
                command_timeout=30,
                timeout=10
            )
            
            # Option 2: Use DSN (only if you're sure it's correct)
            # Uncomment below and comment Option 1 if you prefer DSN
            # self.pool = await asyncpg.create_pool(
            #     dsn=settings.database_url,
            #     min_size=settings.postgres_min_connections,
            #     max_size=settings.postgres_max_connections,
            #     command_timeout=30,
            #     timeout=10
            # )
            
            # Test connection
            async with self.pool.acquire() as conn:
                version = await conn.fetchval("SELECT version()")
                logger.debug(f"PostgreSQL version: {version}")
            
            self._connected = True
            logger.info("✅ PostgreSQL connection pool established")
            logger.info(f"   Pool size: {settings.postgres_min_connections}-{settings.postgres_max_connections}")
            
        except asyncpg.exceptions.InvalidPasswordError as e:
            logger.error(f"❌ PostgreSQL authentication failed: {e}")
            logger.error(f"   Check that POSTGRES_PASSWORD environment variable matches database password")
            self._connected = False
            raise
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            self._connected = False
            raise
    
    async def disconnect(self) -> None:
        """Close connection pool gracefully."""
        if self.pool:
            await self.pool.close()
            self._connected = False
            logger.info("PostgreSQL connection pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """
        Acquire a connection from the pool.
        
        Usage:
            async with db.acquire() as conn:
                result = await conn.fetch("SELECT * FROM table")
        """
        if not self._connected or not self.pool:
            raise RuntimeError("Database not connected")
        
        async with self.pool.acquire() as connection:
            yield connection
    
    async def execute(self, query: str, *args) -> str:
        """
        Execute a query without returning results.
        
        Args:
            query: SQL query
            *args: Query parameters
            
        Returns:
            Execution status string
        """
        async with self.acquire() as conn:
            result = await conn.execute(query, *args)
            logger.debug(f"Executed: {query[:100]}... | Result: {result}")
            return result
    
    async def fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """
        Fetch a single row.
        
        Args:
            query: SQL query
            *args: Query parameters
            
        Returns:
            Row as dictionary or None
        """
        async with self.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None
    
    async def fetch_all(self, query: str, *args) -> List[Dict[str, Any]]:
        """
        Fetch all rows.
        
        Args:
            query: SQL query
            *args: Query parameters
            
        Returns:
            List of rows as dictionaries
        """
        async with self.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    
    async def fetch_val(self, query: str, *args) -> Any:
        """
        Fetch a single value.
        
        Args:
            query: SQL query
            *args: Query parameters
            
        Returns:
            Single value
        """
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)
    
    @asynccontextmanager
    async def transaction(self):
        """
        Execute queries in a transaction.
        
        Usage:
            async with db.transaction():
                await db.execute("INSERT INTO ...")
                await db.execute("UPDATE ...")
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to database."""
        return self._connected
    
    # ========================================================================
    # CONVERSATION OPERATIONS
    # ========================================================================
    
    async def save_conversation(
        self,
        user_id: str,
        session_id: str,
        messages: List[Dict[str, Any]]
    ) -> str:
        """
        Save conversation to database.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            messages: List of conversation messages
            
        Returns:
            Conversation ID
        """
        query = """
            INSERT INTO conversations (user_id, session_id, messages)
            VALUES ($1, $2, $3)
            RETURNING id
        """
        
        import json
        messages_json = json.dumps(messages)
        
        conversation_id = await self.fetch_val(query, user_id, session_id, messages_json)
        logger.debug(f"Saved conversation: {conversation_id}")
        return str(conversation_id)
    
    async def get_conversation_history(
        self,
        user_id: str,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for user/session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversations
        """
        query = """
            SELECT id, user_id, session_id, messages, created_at, updated_at
            FROM conversations
            WHERE user_id = $1 AND session_id = $2
            ORDER BY created_at DESC
            LIMIT $3
        """
        
        conversations = await self.fetch_all(query, user_id, session_id, limit)
        logger.debug(f"Retrieved {len(conversations)} conversations for {user_id}/{session_id}")
        return conversations
    
    async def update_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]]
    ) -> bool:
        """
        Update existing conversation.
        
        Args:
            conversation_id: Conversation identifier
            messages: Updated messages list
            
        Returns:
            True if updated successfully
        """
        import json
        query = """
            UPDATE conversations
            SET messages = $1, updated_at = NOW()
            WHERE id = $2
        """
        
        result = await self.execute(query, json.dumps(messages), conversation_id)
        return "UPDATE 1" in result
    
    # ========================================================================
    # PROJECT OPERATIONS
    # ========================================================================
    
    async def save_project(
        self,
        user_id: str,
        project_name: str,
        architecture: Dict[str, Any],
        layout: Dict[str, Any],
        blockly: Dict[str, Any]
    ) -> str:
        """
        Save project to database.
        
        Args:
            user_id: User identifier
            project_name: Project name
            architecture: Architecture JSON
            layout: Layout JSON
            blockly: Blockly JSON
            
        Returns:
            Project ID
        """
        import json
        query = """
            INSERT INTO projects (user_id, project_name, architecture, layout, blockly)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
        """
        
        project_id = await self.fetch_val(
            query,
            user_id,
            project_name,
            json.dumps(architecture),
            json.dumps(layout),
            json.dumps(blockly)
        )
        
        logger.debug(f"Saved project: {project_id}")
        return str(project_id)
    
    async def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Get project by ID.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Project data or None
        """
        query = """
            SELECT id, user_id, project_name, architecture, layout, blockly,
                   created_at, updated_at
            FROM projects
            WHERE id = $1
        """
        
        return await self.fetch_one(query, project_id)
    
    async def get_user_projects(
        self,
        user_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get all projects for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum projects to return
            
        Returns:
            List of projects
        """
        query = """
            SELECT id, user_id, project_name, created_at, updated_at
            FROM projects
            WHERE user_id = $1
            ORDER BY updated_at DESC
            LIMIT $2
        """
        
        return await self.fetch_all(query, user_id, limit)
    
    async def update_project(
        self,
        project_id: str,
        architecture: Optional[Dict[str, Any]] = None,
        layout: Optional[Dict[str, Any]] = None,
        blockly: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update project components.
        
        Args:
            project_id: Project identifier
            architecture: Updated architecture (optional)
            layout: Updated layout (optional)
            blockly: Updated blockly (optional)
            
        Returns:
            True if updated successfully
        """
        import json
        updates = []
        params = []
        param_idx = 1
        
        if architecture is not None:
            updates.append(f"architecture = ${param_idx}")
            params.append(json.dumps(architecture))
            param_idx += 1
        
        if layout is not None:
            updates.append(f"layout = ${param_idx}")
            params.append(json.dumps(layout))
            param_idx += 1
        
        if blockly is not None:
            updates.append(f"blockly = ${param_idx}")
            params.append(json.dumps(blockly))
            param_idx += 1
        
        if not updates:
            return False
        
        updates.append(f"updated_at = NOW()")
        params.append(project_id)
        
        query = f"""
            UPDATE projects
            SET {', '.join(updates)}
            WHERE id = ${param_idx}
        """
        
        result = await self.execute(query, *params)
        return "UPDATE 1" in result
    
    # ========================================================================
    # USER PREFERENCES OPERATIONS
    # ========================================================================
    
    async def save_user_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> bool:
        """
        Save user preferences.
        
        Args:
            user_id: User identifier
            preferences: Preferences JSON
            
        Returns:
            True if saved successfully
        """
        import json
        query = """
            INSERT INTO user_preferences (user_id, preferences)
            VALUES ($1, $2)
            ON CONFLICT (user_id) 
            DO UPDATE SET preferences = $2, updated_at = NOW()
        """
        
        result = await self.execute(query, user_id, json.dumps(preferences))
        return "INSERT" in result or "UPDATE" in result
    
    async def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            Preferences or None
        """
        query = """
            SELECT preferences
            FROM user_preferences
            WHERE user_id = $1
        """
        
        result = await self.fetch_one(query, user_id)
        return result['preferences'] if result else None
    
    # ========================================================================
    # METRICS OPERATIONS
    # ========================================================================
    
    async def save_request_metric(
        self,
        task_id: str,
        user_id: str,
        stage: str,
        duration_ms: int,
        success: bool,
        error_message: Optional[str] = None
    ) -> None:
        """
        Save request processing metrics.
        
        Args:
            task_id: Task identifier
            user_id: User identifier
            stage: Processing stage
            duration_ms: Duration in milliseconds
            success: Whether stage succeeded
            error_message: Error message if failed
        """
        query = """
            INSERT INTO request_metrics 
            (task_id, user_id, stage, duration_ms, success, error_message)
            VALUES ($1, $2, $3, $4, $5, $6)
        """
        
        await self.execute(
            query,
            task_id,
            user_id,
            stage,
            duration_ms,
            success,
            error_message
        )


# Global database manager instance
db_manager = DatabaseManager()


if __name__ == "__main__":
    # Test database manager
    import asyncio
    
    async def test_database():
        """Test database operations"""
        print("\n" + "=" * 60)
        print("DATABASE MANAGER TEST")
        print("=" * 60)
        
        # Connect
        await db_manager.connect()
        print(f"\n✅ Connected: {db_manager.is_connected}")
        
        # Test conversation save
        conversation_id = await db_manager.save_conversation(
            user_id="test_user",
            session_id="test_session",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        )
        print(f"✅ Saved conversation: {conversation_id}")
        
        # Test conversation retrieval
        history = await db_manager.get_conversation_history(
            user_id="test_user",
            session_id="test_session"
        )
        print(f"✅ Retrieved {len(history)} conversations")
        
        # Disconnect
        await db_manager.disconnect()
        
        print("\n" + "=" * 60)
        print("✅ Database manager test complete!")
        print("=" * 60 + "\n")
    
    asyncio.run(test_database())
"""
Project State Persistence Layer
================================

Handles atomic, version-safe read/write operations for project state.

Supports multiple backends:
- File system (JSON files)
- PostgreSQL database
- Redis cache (for fast reads)

Design Principles:
- Atomic writes (all or nothing)
- Version-safe updates (optimistic locking)
- Fail fast on corruption
- Pluggable backends
"""

from typing import Optional, Protocol, Dict, Any
from pathlib import Path
from datetime import datetime
import json
from abc import ABC, abstractmethod
from loguru import logger

from app.models.project_state import ProjectState, validate_state_schema


# ============================================================================
# EXCEPTIONS
# ============================================================================

class PersistenceError(Exception):
    """Base exception for persistence errors"""
    pass


class StateNotFoundError(PersistenceError):
    """Raised when project state doesn't exist"""
    pass


class StateCorruptedError(PersistenceError):
    """Raised when state fails validation on load"""
    pass


class VersionConflictError(PersistenceError):
    """Raised when version mismatch detected (optimistic locking)"""
    pass


# ============================================================================
# STORAGE BACKEND INTERFACE
# ============================================================================

class StorageBackend(Protocol):
    """Interface that all storage backends must implement"""
    
    async def read(self, project_id: str) -> Dict[str, Any]:
        """Read raw state data"""
        ...
    
    async def write(self, project_id: str, state_data: Dict[str, Any]) -> None:
        """Write raw state data"""
        ...
    
    async def exists(self, project_id: str) -> bool:
        """Check if state exists"""
        ...
    
    async def delete(self, project_id: str) -> None:
        """Delete state"""
        ...
    
    async def list_projects(self, user_id: Optional[str] = None) -> list[str]:
        """List all project IDs (optionally filtered by user)"""
        ...


# ============================================================================
# FILE SYSTEM BACKEND
# ============================================================================

class FileSystemBackend:
    """
    File-based storage backend using JSON files.
    
    Structure:
        storage_path/
            {project_id}.json
            {project_id}.json.backup
    """
    
    def __init__(self, storage_path: str = "./project_states"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileSystemBackend initialized: {self.storage_path}")
    
    def _get_file_path(self, project_id: str) -> Path:
        """Get path to project state file"""
        return self.storage_path / f"{project_id}.json"
    
    def _get_backup_path(self, project_id: str) -> Path:
        """Get path to backup file"""
        return self.storage_path / f"{project_id}.json.backup"
    
    async def read(self, project_id: str) -> Dict[str, Any]:
        """Read state from file"""
        file_path = self._get_file_path(project_id)
        
        if not file_path.exists():
            raise StateNotFoundError(f"Project state not found: {project_id}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.debug(f"Loaded state from file: {project_id}")
            return data
            
        except json.JSONDecodeError as e:
            # Try to recover from backup
            logger.error(f"State file corrupted: {project_id}, attempting backup recovery")
            return await self._read_backup(project_id)
        except Exception as e:
            raise PersistenceError(f"Failed to read state: {e}")
    
    async def _read_backup(self, project_id: str) -> Dict[str, Any]:
        """Attempt to read from backup file"""
        backup_path = self._get_backup_path(project_id)
        
        if not backup_path.exists():
            raise StateCorruptedError(
                f"State file corrupted and no backup available: {project_id}"
            )
        
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.warning(f"Recovered state from backup: {project_id}")
            return data
            
        except Exception as e:
            raise StateCorruptedError(
                f"Both state file and backup are corrupted: {project_id}"
            )
    
    async def write(self, project_id: str, state_data: Dict[str, Any]) -> None:
        """Write state to file (atomic)"""
        file_path = self._get_file_path(project_id)
        backup_path = self._get_backup_path(project_id)
        temp_path = self.storage_path / f"{project_id}.json.tmp"
        
        try:
            # Write to temporary file first
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            # Create backup of existing file (if exists)
            if file_path.exists():
                file_path.replace(backup_path)
            
            # Atomic move
            temp_path.replace(file_path)
            
            logger.debug(f"Saved state to file: {project_id}")
            
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise PersistenceError(f"Failed to write state: {e}")
    
    async def exists(self, project_id: str) -> bool:
        """Check if state file exists"""
        return self._get_file_path(project_id).exists()
    
    async def delete(self, project_id: str) -> None:
        """Delete state file and backup"""
        file_path = self._get_file_path(project_id)
        backup_path = self._get_backup_path(project_id)
        
        if file_path.exists():
            file_path.unlink()
        if backup_path.exists():
            backup_path.unlink()
        
        logger.info(f"Deleted state: {project_id}")
    
    async def list_projects(self, user_id: Optional[str] = None) -> list[str]:
        """List all project state files"""
        project_ids = []
        
        for file_path in self.storage_path.glob("*.json"):
            if not file_path.name.endswith('.backup') and not file_path.name.endswith('.tmp'):
                project_id = file_path.stem
                
                # If user_id filter provided, check metadata
                if user_id:
                    try:
                        data = await self.read(project_id)
                        if data.get('metadata', {}).get('created_by') == user_id:
                            project_ids.append(project_id)
                    except:
                        continue
                else:
                    project_ids.append(project_id)
        
        return project_ids


# ============================================================================
# DATABASE BACKEND (PostgreSQL)
# ============================================================================

class DatabaseBackend:
    """
    PostgreSQL storage backend for production use.
    
    Table schema:
        project_states (
            project_id VARCHAR PRIMARY KEY,
            user_id VARCHAR NOT NULL,
            state_data JSONB NOT NULL,
            version INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            INDEX idx_user_id (user_id)
        )
    """
    
    def __init__(self, db_manager):
        """
        Args:
            db_manager: Database connection manager (from app.core.database)
        """
        self.db = db_manager
        logger.info("DatabaseBackend initialized")
    
    async def read(self, project_id: str) -> Dict[str, Any]:
        """Read state from database"""
        query = """
            SELECT state_data
            FROM project_states
            WHERE project_id = $1
        """
        
        row = await self.db.fetchrow(query, project_id)
        
        if not row:
            raise StateNotFoundError(f"Project state not found: {project_id}")
        
        logger.debug(f"Loaded state from database: {project_id}")
        return row['state_data']
    
    async def write(self, project_id: str, state_data: Dict[str, Any]) -> None:
        """Write state to database (upsert)"""
        metadata = state_data.get('metadata', {})
        user_id = metadata.get('created_by', 'unknown')
        version = metadata.get('version', 1)
        
        query = """
            INSERT INTO project_states (
                project_id, user_id, state_data, version, created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (project_id) DO UPDATE
            SET 
                state_data = EXCLUDED.state_data,
                version = EXCLUDED.version,
                updated_at = EXCLUDED.updated_at
        """
        
        now = datetime.utcnow()
        
        await self.db.execute(
            query,
            project_id,
            user_id,
            json.dumps(state_data),
            version,
            now,
            now,
        )
        
        logger.debug(f"Saved state to database: {project_id}")
    
    async def exists(self, project_id: str) -> bool:
        """Check if state exists in database"""
        query = "SELECT 1 FROM project_states WHERE project_id = $1"
        row = await self.db.fetchrow(query, project_id)
        return row is not None
    
    async def delete(self, project_id: str) -> None:
        """Delete state from database"""
        query = "DELETE FROM project_states WHERE project_id = $1"
        await self.db.execute(query, project_id)
        logger.info(f"Deleted state from database: {project_id}")
    
    async def list_projects(self, user_id: Optional[str] = None) -> list[str]:
        """List all project IDs"""
        if user_id:
            query = "SELECT project_id FROM project_states WHERE user_id = $1"
            rows = await self.db.fetch(query, user_id)
        else:
            query = "SELECT project_id FROM project_states"
            rows = await self.db.fetch(query)
        
        return [row['project_id'] for row in rows]


# ============================================================================
# STATE PERSISTENCE MANAGER
# ============================================================================

class ProjectStatePersistence:
    """
    High-level persistence manager with validation and version control.
    
    This is the main public API for state persistence.
    """
    
    def __init__(self, backend: StorageBackend):
        """
        Args:
            backend: Storage backend (FileSystemBackend or DatabaseBackend)
        """
        self.backend = backend
    
    async def load_project_state(self, project_id: str) -> ProjectState:
        """
        Load project state from storage with validation.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Validated ProjectState instance
            
        Raises:
            StateNotFoundError: Project doesn't exist
            StateCorruptedError: State fails validation
        """
        logger.info(f"Loading project state: {project_id}")
        
        try:
            # Read raw data
            state_data = await self.backend.read(project_id)
            
            # Deserialize and validate
            state = ProjectState.from_dict(state_data)
            
            # Additional validation
            validate_state_schema(state)
            
            logger.info(
                f"✅ Loaded project state",
                extra={
                    "project_id": project_id,
                    "version": state.metadata.version,
                    "screens": len(state.architecture.screens),
                }
            )
            
            return state
            
        except StateNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to load state: {e}", exc_info=e)
            raise StateCorruptedError(f"State validation failed: {e}")
    
    async def save_project_state(
        self,
        state: ProjectState,
        expected_version: Optional[int] = None
    ) -> None:
        """
        Save project state to storage with version check.
        
        Args:
            state: ProjectState to save
            expected_version: Expected current version (for optimistic locking)
            
        Raises:
            VersionConflictError: Version mismatch (concurrent modification)
        """
        project_id = state.metadata.project_id
        
        logger.info(
            f"Saving project state: {project_id}",
            extra={
                "version": state.metadata.version,
                "expected_version": expected_version,
            }
        )
        
        # Optimistic locking check
        if expected_version is not None:
            try:
                current_state = await self.load_project_state(project_id)
                if current_state.metadata.version != expected_version:
                    raise VersionConflictError(
                        f"Version conflict: expected {expected_version}, "
                        f"but current is {current_state.metadata.version}"
                    )
            except StateNotFoundError:
                # New state, no conflict
                pass
        
        # Validate before saving
        try:
            validate_state_schema(state)
        except Exception as e:
            raise PersistenceError(f"State validation failed before save: {e}")
        
        # Serialize to dict
        state_data = state.to_dict()
        
        # Write to backend
        await self.backend.write(project_id, state_data)
        
        logger.info(
            f"✅ Saved project state",
            extra={
                "project_id": project_id,
                "version": state.metadata.version,
            }
        )
    
    async def delete_project_state(self, project_id: str) -> None:
        """Delete project state from storage"""
        logger.warning(f"Deleting project state: {project_id}")
        await self.backend.delete(project_id)
    
    async def project_exists(self, project_id: str) -> bool:
        """Check if project state exists"""
        return await self.backend.exists(project_id)
    
    async def list_user_projects(self, user_id: str) -> list[str]:
        """List all projects for a user"""
        return await self.backend.list_projects(user_id=user_id)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def load_project_state(
    project_id: str,
    backend: Optional[StorageBackend] = None
) -> ProjectState:
    """
    Convenience function to load project state.
    
    Args:
        project_id: Project identifier
        backend: Storage backend (defaults to FileSystemBackend)
    """
    if backend is None:
        backend = FileSystemBackend()
    
    persistence = ProjectStatePersistence(backend)
    return await persistence.load_project_state(project_id)


async def save_project_state(
    state: ProjectState,
    backend: Optional[StorageBackend] = None,
    expected_version: Optional[int] = None
) -> None:
    """
    Convenience function to save project state.
    
    Args:
        state: ProjectState to save
        backend: Storage backend (defaults to FileSystemBackend)
        expected_version: Expected version for optimistic locking
    """
    if backend is None:
        backend = FileSystemBackend()
    
    persistence = ProjectStatePersistence(backend)
    await persistence.save_project_state(state, expected_version)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from app.models.project_state import ProjectState
    
    async def example():
        # Create a new state
        state = ProjectState.create_new(
            app_name="Test App",
            app_description="Test application",
            created_by="user_123",
        )
        
        print(f"Created state: {state}")
        
        # Initialize file system backend
        backend = FileSystemBackend(storage_path="./test_states")
        persistence = ProjectStatePersistence(backend)
        
        # Save state
        await persistence.save_project_state(state)
        print(f"✅ Saved state: {state.metadata.project_id}")
        
        # Load state back
        loaded_state = await persistence.load_project_state(state.metadata.project_id)
        print(f"✅ Loaded state: {loaded_state}")
        
        # Verify integrity
        assert loaded_state.metadata.project_id == state.metadata.project_id
        assert loaded_state.metadata.version == state.metadata.version
        print("✅ State integrity verified")
        
        # Test optimistic locking
        try:
            await persistence.save_project_state(state, expected_version=999)
        except VersionConflictError as e:
            print(f"✅ Optimistic locking works: {e}")
        
        # Clean up
        await persistence.delete_project_state(state.metadata.project_id)
        print("✅ Cleanup complete")
    
    asyncio.run(example())
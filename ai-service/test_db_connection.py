"""
Database Connection Diagnostic Script
Run this to verify your environment variables are loaded correctly
"""
import os
import asyncio
import asyncpg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_env_vars():
    """Check all database-related environment variables"""
    print("\n" + "=" * 70)
    print("ENVIRONMENT VARIABLES CHECK")
    print("=" * 70)
    
    vars_to_check = [
        "POSTGRES_HOST",
        "POSTGRES_PORT", 
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "DATABASE_URL"
    ]
    
    for var in vars_to_check:
        value = os.getenv(var)
        if var == "POSTGRES_PASSWORD":
            # Mask password but show length
            display = f"{'*' * len(value)} (length: {len(value)})" if value else "NOT SET"
        else:
            display = value if value else "NOT SET"
        
        status = "✅" if value else "❌"
        print(f"{status} {var:25s} = {display}")
    
    print("=" * 70 + "\n")


async def test_connection_direct():
    """Test connection using direct parameters"""
    print("=" * 70)
    print("TEST 1: Direct Parameters Connection")
    print("=" * 70)
    
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    database = os.getenv("POSTGRES_DB", "appdb")
    user = os.getenv("POSTGRES_USER", "admin")
    password = os.getenv("POSTGRES_PASSWORD", "devpass")
    
    print(f"Host:     {host}")
    print(f"Port:     {port}")
    print(f"Database: {database}")
    print(f"User:     {user}")
    print(f"Password: {'*' * len(password)} (length: {len(password)})")
    print()
    
    try:
        conn = await asyncpg.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            timeout=10
        )
        
        version = await conn.fetchval("SELECT version()")
        print(f"✅ Connection successful!")
        print(f"   PostgreSQL version: {version[:50]}...")
        
        await conn.close()
        return True
        
    except asyncpg.exceptions.InvalidPasswordError as e:
        print(f"❌ Authentication failed: {e}")
        print(f"\n💡 The password '{password}' is not correct for user '{user}'")
        print(f"   Try resetting the password with:")
        print(f"   docker exec -it local-postgres psql -U postgres -c \"ALTER USER {user} WITH PASSWORD '{password}';\"")
        return False
        
    except asyncpg.exceptions.InvalidCatalogNameError:
        print(f"❌ Database '{database}' does not exist")
        print(f"   Create it with:")
        print(f"   docker exec -it local-postgres psql -U postgres -c \"CREATE DATABASE {database};\"")
        return False
        
    except Exception as e:
        print(f"❌ Connection failed: {type(e).__name__}: {e}")
        return False


async def test_connection_dsn():
    """Test connection using DATABASE_URL"""
    print("\n" + "=" * 70)
    print("TEST 2: DSN Connection (DATABASE_URL)")
    print("=" * 70)
    
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        print("❌ DATABASE_URL not set")
        return False
    
    # Mask password in URL for display
    display_url = database_url
    if "@" in database_url and ":" in database_url.split("@")[0]:
        parts = database_url.split("://")[1].split("@")
        user_pass = parts[0].split(":")
        masked = f"{user_pass[0]}:****@{parts[1]}"
        display_url = f"{database_url.split('://')[0]}://{masked}"
    
    print(f"DATABASE_URL: {display_url}")
    print()
    
    try:
        conn = await asyncpg.connect(dsn=database_url, timeout=10)
        
        version = await conn.fetchval("SELECT version()")
        print(f"✅ Connection successful!")
        print(f"   PostgreSQL version: {version[:50]}...")
        
        await conn.close()
        return True
        
    except asyncpg.exceptions.InvalidPasswordError as e:
        print(f"❌ Authentication failed: {e}")
        print(f"\n💡 Check that DATABASE_URL password matches the actual password")
        return False
        
    except Exception as e:
        print(f"❌ Connection failed: {type(e).__name__}: {e}")
        return False


async def test_connection_pool():
    """Test connection pool creation"""
    print("\n" + "=" * 70)
    print("TEST 3: Connection Pool")
    print("=" * 70)
    
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    database = os.getenv("POSTGRES_DB", "appdb")
    user = os.getenv("POSTGRES_USER", "admin")
    password = os.getenv("POSTGRES_PASSWORD", "devpass")
    
    try:
        pool = await asyncpg.create_pool(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            min_size=2,
            max_size=10,
            timeout=10
        )
        
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            print(f"✅ Connection pool created successfully!")
            print(f"   Test query result: {result}")
        
        await pool.close()
        return True
        
    except Exception as e:
        print(f"❌ Pool creation failed: {type(e).__name__}: {e}")
        return False


async def main():
    """Run all diagnostic tests"""
    print("\n" + "=" * 70)
    print("PostgreSQL CONNECTION DIAGNOSTICS")
    print("=" * 70 + "\n")
    
    # Check environment variables
    check_env_vars()
    
    # Test connections
    results = []
    results.append(await test_connection_direct())
    results.append(await test_connection_dsn())
    results.append(await test_connection_pool())
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All tests passed ({passed}/{total})")
        print("\n🎉 Your database connection is properly configured!")
    else:
        print(f"❌ Some tests failed ({passed}/{total} passed)")
        print("\n💡 Fix the issues above and run this script again")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
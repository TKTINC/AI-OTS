#!/usr/bin/env python3
"""
Database migration script for AI Options Trading System
Handles schema creation, updates, and data migrations
"""

import os
import sys
import psycopg2
import logging
from datetime import datetime
from typing import List, Dict, Any
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """Handles database migrations for the trading system"""
    
    def __init__(self, database_url: str):
        """Initialize the migrator with database connection"""
        self.database_url = database_url
        self.connection = None
        self.migration_files = [
            '001_initial_schema.sql',
            '002_hypertables.sql',
            '003_indexes.sql',
            '004_retention_policies.sql'
        ]
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(self.database_url)
            self.connection.autocommit = True
            logger.info("Connected to database successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from database")
    
    def create_migration_table(self):
        """Create migration tracking table"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS migration_history (
                        id SERIAL PRIMARY KEY,
                        filename VARCHAR(255) NOT NULL UNIQUE,
                        applied_at TIMESTAMPTZ DEFAULT NOW(),
                        checksum VARCHAR(64),
                        success BOOLEAN DEFAULT TRUE,
                        error_message TEXT
                    );
                """)
            logger.info("Migration tracking table created/verified")
        except Exception as e:
            logger.error(f"Failed to create migration table: {e}")
            raise
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of already applied migrations"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT filename FROM migration_history WHERE success = TRUE"
                )
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get applied migrations: {e}")
            return []
    
    def calculate_checksum(self, content: str) -> str:
        """Calculate MD5 checksum of migration content"""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()
    
    def read_migration_file(self, filename: str) -> str:
        """Read migration file content"""
        schema_dir = os.path.join(os.path.dirname(__file__), '..', 'schema')
        file_path = os.path.join(schema_dir, filename)
        
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Migration file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to read migration file {filename}: {e}")
            raise
    
    def apply_migration(self, filename: str) -> bool:
        """Apply a single migration file"""
        logger.info(f"Applying migration: {filename}")
        
        try:
            # Read migration content
            content = self.read_migration_file(filename)
            checksum = self.calculate_checksum(content)
            
            # Execute migration
            with self.connection.cursor() as cursor:
                cursor.execute(content)
            
            # Record successful migration
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO migration_history (filename, checksum, success)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (filename) DO UPDATE SET
                        applied_at = NOW(),
                        checksum = EXCLUDED.checksum,
                        success = EXCLUDED.success,
                        error_message = NULL
                """, (filename, checksum, True))
            
            logger.info(f"Successfully applied migration: {filename}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to apply migration {filename}: {error_msg}")
            
            # Record failed migration
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO migration_history (filename, success, error_message)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (filename) DO UPDATE SET
                            applied_at = NOW(),
                            success = EXCLUDED.success,
                            error_message = EXCLUDED.error_message
                    """, (filename, False, error_msg))
            except:
                pass  # Don't fail if we can't record the error
            
            return False
    
    def run_migrations(self, force: bool = False) -> bool:
        """Run all pending migrations"""
        logger.info("Starting database migrations")
        
        try:
            self.connect()
            self.create_migration_table()
            
            applied_migrations = self.get_applied_migrations()
            pending_migrations = []
            
            for migration_file in self.migration_files:
                if migration_file not in applied_migrations or force:
                    pending_migrations.append(migration_file)
            
            if not pending_migrations:
                logger.info("No pending migrations")
                return True
            
            logger.info(f"Found {len(pending_migrations)} pending migrations")
            
            success_count = 0
            for migration_file in pending_migrations:
                if self.apply_migration(migration_file):
                    success_count += 1
                else:
                    logger.error(f"Migration failed: {migration_file}")
                    if not force:
                        break
            
            if success_count == len(pending_migrations):
                logger.info("All migrations completed successfully")
                return True
            else:
                logger.error(f"Only {success_count}/{len(pending_migrations)} migrations succeeded")
                return False
                
        except Exception as e:
            logger.error(f"Migration process failed: {e}")
            return False
        finally:
            self.disconnect()
    
    def rollback_migration(self, filename: str) -> bool:
        """Mark a migration as not applied (for manual rollback)"""
        logger.info(f"Rolling back migration: {filename}")
        
        try:
            self.connect()
            
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM migration_history WHERE filename = %s",
                    (filename,)
                )
                
                if cursor.rowcount > 0:
                    logger.info(f"Migration {filename} marked as not applied")
                    return True
                else:
                    logger.warning(f"Migration {filename} was not found in history")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to rollback migration {filename}: {e}")
            return False
        finally:
            self.disconnect()
    
    def get_migration_status(self) -> List[Dict[str, Any]]:
        """Get status of all migrations"""
        try:
            self.connect()
            
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT filename, applied_at, success, error_message
                    FROM migration_history
                    ORDER BY applied_at
                """)
                
                results = []
                applied_migrations = {row[0]: row for row in cursor.fetchall()}
                
                for migration_file in self.migration_files:
                    if migration_file in applied_migrations:
                        row = applied_migrations[migration_file]
                        results.append({
                            'filename': row[0],
                            'status': 'SUCCESS' if row[2] else 'FAILED',
                            'applied_at': row[1],
                            'error_message': row[3]
                        })
                    else:
                        results.append({
                            'filename': migration_file,
                            'status': 'PENDING',
                            'applied_at': None,
                            'error_message': None
                        })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get migration status: {e}")
            return []
        finally:
            self.disconnect()
    
    def verify_database_health(self) -> bool:
        """Verify database health and TimescaleDB functionality"""
        logger.info("Verifying database health")
        
        try:
            self.connect()
            
            with self.connection.cursor() as cursor:
                # Check TimescaleDB extension
                cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
                if not cursor.fetchone():
                    logger.error("TimescaleDB extension not found")
                    return False
                
                # Check if hypertables exist
                cursor.execute("""
                    SELECT schemaname, tablename 
                    FROM timescaledb_information.hypertables 
                    WHERE schemaname = 'trading'
                """)
                hypertables = cursor.fetchall()
                
                expected_hypertables = [
                    'stock_prices', 'options_data', 'signals', 
                    'performance_metrics', 'data_quality', 'system_events'
                ]
                
                found_hypertables = [row[1] for row in hypertables]
                missing_hypertables = set(expected_hypertables) - set(found_hypertables)
                
                if missing_hypertables:
                    logger.error(f"Missing hypertables: {missing_hypertables}")
                    return False
                
                # Check continuous aggregates
                cursor.execute("""
                    SELECT view_name 
                    FROM timescaledb_information.continuous_aggregates
                    WHERE view_schema = 'trading'
                """)
                continuous_aggs = [row[0] for row in cursor.fetchall()]
                
                expected_aggs = [
                    'stock_prices_1m', 'stock_prices_5m', 'stock_prices_1h', 
                    'stock_prices_1d', 'options_volume_1h', 'signal_performance_1d'
                ]
                
                missing_aggs = set(expected_aggs) - set(continuous_aggs)
                if missing_aggs:
                    logger.warning(f"Missing continuous aggregates: {missing_aggs}")
                
                logger.info("Database health check passed")
                return True
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
        finally:
            self.disconnect()

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Database migration tool for AI-OTS')
    parser.add_argument('--database-url', required=True, help='Database connection URL')
    parser.add_argument('--action', choices=['migrate', 'status', 'rollback', 'health'], 
                       default='migrate', help='Action to perform')
    parser.add_argument('--force', action='store_true', 
                       help='Force re-apply all migrations')
    parser.add_argument('--migration', help='Specific migration file for rollback')
    
    args = parser.parse_args()
    
    migrator = DatabaseMigrator(args.database_url)
    
    if args.action == 'migrate':
        success = migrator.run_migrations(force=args.force)
        sys.exit(0 if success else 1)
    
    elif args.action == 'status':
        status = migrator.get_migration_status()
        print("\nMigration Status:")
        print("-" * 60)
        for migration in status:
            print(f"{migration['filename']:<30} {migration['status']:<10} {migration['applied_at'] or 'N/A'}")
        
    elif args.action == 'rollback':
        if not args.migration:
            print("Error: --migration required for rollback action")
            sys.exit(1)
        success = migrator.rollback_migration(args.migration)
        sys.exit(0 if success else 1)
    
    elif args.action == 'health':
        success = migrator.verify_database_health()
        sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()


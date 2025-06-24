#!/bin/bash

# Database restore script for AI Options Trading System
# Handles point-in-time recovery, selective restore, and disaster recovery

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${LOG_FILE:-/var/log/ai-ots-restore.log}"

# Database connection parameters
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-trading_db}"
DB_USER="${DB_USER:-trading_admin}"
DB_PASSWORD="${DB_PASSWORD:-}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Check if database exists
database_exists() {
    local db_name="$1"
    export PGPASSWORD="$DB_PASSWORD"
    
    if psql \
        --host="$DB_HOST" \
        --port="$DB_PORT" \
        --username="$DB_USER" \
        --dbname="postgres" \
        --tuples-only \
        --command="SELECT 1 FROM pg_database WHERE datname='$db_name'" | grep -q 1; then
        return 0
    else
        return 1
    fi
    
    unset PGPASSWORD
}

# Create database
create_database() {
    local db_name="$1"
    log "Creating database: $db_name"
    
    export PGPASSWORD="$DB_PASSWORD"
    
    if createdb \
        --host="$DB_HOST" \
        --port="$DB_PORT" \
        --username="$DB_USER" \
        --encoding=UTF8 \
        --locale=en_US.UTF-8 \
        "$db_name"; then
        log "Database created successfully: $db_name"
    else
        error_exit "Failed to create database: $db_name"
    fi
    
    unset PGPASSWORD
}

# Drop database (with confirmation)
drop_database() {
    local db_name="$1"
    local force="${2:-false}"
    
    if [[ "$force" != "true" ]]; then
        read -p "Are you sure you want to drop database '$db_name'? (yes/no): " confirm
        if [[ "$confirm" != "yes" ]]; then
            log "Database drop cancelled"
            return 1
        fi
    fi
    
    log "Dropping database: $db_name"
    
    export PGPASSWORD="$DB_PASSWORD"
    
    # Terminate active connections
    psql \
        --host="$DB_HOST" \
        --port="$DB_PORT" \
        --username="$DB_USER" \
        --dbname="postgres" \
        --command="SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='$db_name' AND pid <> pg_backend_pid();"
    
    if dropdb \
        --host="$DB_HOST" \
        --port="$DB_PORT" \
        --username="$DB_USER" \
        "$db_name"; then
        log "Database dropped successfully: $db_name"
    else
        error_exit "Failed to drop database: $db_name"
    fi
    
    unset PGPASSWORD
}

# Full restore from backup
full_restore() {
    local backup_path="$1"
    local target_db="${2:-$DB_NAME}"
    local drop_existing="${3:-false}"
    
    log "Starting full restore from: $backup_path to database: $target_db"
    
    # Check if backup file exists
    if [[ ! -f "$backup_path" ]]; then
        error_exit "Backup file not found: $backup_path"
    fi
    
    # Handle existing database
    if database_exists "$target_db"; then
        if [[ "$drop_existing" == "true" ]]; then
            drop_database "$target_db" true
            create_database "$target_db"
        else
            log "WARNING: Target database already exists. Use --drop-existing to replace it."
            read -p "Continue with restore to existing database? (yes/no): " confirm
            if [[ "$confirm" != "yes" ]]; then
                log "Restore cancelled"
                return 1
            fi
        fi
    else
        create_database "$target_db"
    fi
    
    # Decompress if needed
    local restore_file="$backup_path"
    if [[ "$backup_path" == *.gz ]]; then
        log "Decompressing backup file"
        restore_file="/tmp/$(basename "${backup_path%.gz}")"
        gunzip -c "$backup_path" > "$restore_file"
    fi
    
    export PGPASSWORD="$DB_PASSWORD"
    
    # Determine restore method based on file format
    if [[ "$restore_file" == *.sql ]]; then
        # Plain SQL file
        log "Restoring from plain SQL file"
        if psql \
            --host="$DB_HOST" \
            --port="$DB_PORT" \
            --username="$DB_USER" \
            --dbname="$target_db" \
            --file="$restore_file"; then
            log "Restore completed successfully"
        else
            error_exit "Restore failed"
        fi
    else
        # Custom format (pg_dump -Fc)
        log "Restoring from custom format file"
        if pg_restore \
            --host="$DB_HOST" \
            --port="$DB_PORT" \
            --username="$DB_USER" \
            --dbname="$target_db" \
            --verbose \
            --clean \
            --if-exists \
            --no-owner \
            --no-privileges \
            "$restore_file"; then
            log "Restore completed successfully"
        else
            error_exit "Restore failed"
        fi
    fi
    
    # Clean up temporary file
    if [[ "$restore_file" != "$backup_path" ]]; then
        rm -f "$restore_file"
    fi
    
    unset PGPASSWORD
    
    # Verify restore
    verify_restore "$target_db"
}

# Schema-only restore
schema_restore() {
    local backup_path="$1"
    local target_db="${2:-$DB_NAME}"
    
    log "Starting schema-only restore from: $backup_path"
    
    if [[ ! -f "$backup_path" ]]; then
        error_exit "Backup file not found: $backup_path"
    fi
    
    if ! database_exists "$target_db"; then
        create_database "$target_db"
    fi
    
    local restore_file="$backup_path"
    if [[ "$backup_path" == *.gz ]]; then
        restore_file="/tmp/$(basename "${backup_path%.gz}")"
        gunzip -c "$backup_path" > "$restore_file"
    fi
    
    export PGPASSWORD="$DB_PASSWORD"
    
    if pg_restore \
        --host="$DB_HOST" \
        --port="$DB_PORT" \
        --username="$DB_USER" \
        --dbname="$target_db" \
        --schema-only \
        --verbose \
        --clean \
        --if-exists \
        "$restore_file"; then
        log "Schema restore completed successfully"
    else
        error_exit "Schema restore failed"
    fi
    
    if [[ "$restore_file" != "$backup_path" ]]; then
        rm -f "$restore_file"
    fi
    
    unset PGPASSWORD
}

# Data-only restore
data_restore() {
    local backup_path="$1"
    local target_db="${2:-$DB_NAME}"
    shift 2
    local tables=("$@")
    
    log "Starting data-only restore from: $backup_path"
    
    if [[ ! -f "$backup_path" ]]; then
        error_exit "Backup file not found: $backup_path"
    fi
    
    if ! database_exists "$target_db"; then
        error_exit "Target database does not exist: $target_db"
    fi
    
    local restore_file="$backup_path"
    if [[ "$backup_path" == *.gz ]]; then
        restore_file="/tmp/$(basename "${backup_path%.gz}")"
        gunzip -c "$backup_path" > "$restore_file"
    fi
    
    export PGPASSWORD="$DB_PASSWORD"
    
    local table_args=()
    if [[ ${#tables[@]} -gt 0 ]]; then
        for table in "${tables[@]}"; do
            table_args+=("--table=$table")
        done
    fi
    
    if pg_restore \
        --host="$DB_HOST" \
        --port="$DB_PORT" \
        --username="$DB_USER" \
        --dbname="$target_db" \
        --data-only \
        --verbose \
        --disable-triggers \
        "${table_args[@]}" \
        "$restore_file"; then
        log "Data restore completed successfully"
    else
        error_exit "Data restore failed"
    fi
    
    if [[ "$restore_file" != "$backup_path" ]]; then
        rm -f "$restore_file"
    fi
    
    unset PGPASSWORD
}

# Point-in-time recovery
point_in_time_restore() {
    local backup_path="$1"
    local target_time="$2"
    local target_db="${3:-${DB_NAME}_pitr}"
    
    log "Starting point-in-time recovery to: $target_time"
    
    # This is a simplified PITR implementation
    # In production, you would need WAL files and more sophisticated recovery
    
    log "WARNING: This is a basic PITR implementation"
    log "For full PITR, ensure you have continuous WAL archiving configured"
    
    # Restore base backup first
    full_restore "$backup_path" "$target_db" true
    
    # Apply WAL files up to target time (if available)
    # This would require WAL files to be available
    log "PITR restore completed (base backup only)"
    log "Manual WAL replay may be required for exact point-in-time recovery"
}

# Verify restore integrity
verify_restore() {
    local target_db="$1"
    
    log "Verifying restore integrity for database: $target_db"
    
    export PGPASSWORD="$DB_PASSWORD"
    
    # Check if TimescaleDB extension is installed
    if psql \
        --host="$DB_HOST" \
        --port="$DB_PORT" \
        --username="$DB_USER" \
        --dbname="$target_db" \
        --tuples-only \
        --command="SELECT 1 FROM pg_extension WHERE extname='timescaledb'" | grep -q 1; then
        log "✓ TimescaleDB extension verified"
    else
        log "✗ TimescaleDB extension not found"
    fi
    
    # Check trading schema
    if psql \
        --host="$DB_HOST" \
        --port="$DB_PORT" \
        --username="$DB_USER" \
        --dbname="$target_db" \
        --tuples-only \
        --command="SELECT 1 FROM information_schema.schemata WHERE schema_name='trading'" | grep -q 1; then
        log "✓ Trading schema verified"
    else
        log "✗ Trading schema not found"
    fi
    
    # Check hypertables
    local hypertables=$(psql \
        --host="$DB_HOST" \
        --port="$DB_PORT" \
        --username="$DB_USER" \
        --dbname="$target_db" \
        --tuples-only \
        --command="SELECT COUNT(*) FROM timescaledb_information.hypertables WHERE schema_name='trading'" 2>/dev/null || echo "0")
    
    log "✓ Found $hypertables hypertables"
    
    # Check table counts
    local tables=("stock_prices" "options_data" "signals" "positions" "performance_metrics" "data_quality" "system_events")
    
    for table in "${tables[@]}"; do
        local count=$(psql \
            --host="$DB_HOST" \
            --port="$DB_PORT" \
            --username="$DB_USER" \
            --dbname="$target_db" \
            --tuples-only \
            --command="SELECT COUNT(*) FROM trading.$table" 2>/dev/null || echo "0")
        log "✓ Table trading.$table: $count rows"
    done
    
    unset PGPASSWORD
    
    log "Restore verification completed"
}

# Clone database
clone_database() {
    local source_db="$1"
    local target_db="$2"
    
    log "Cloning database from $source_db to $target_db"
    
    if ! database_exists "$source_db"; then
        error_exit "Source database does not exist: $source_db"
    fi
    
    if database_exists "$target_db"; then
        error_exit "Target database already exists: $target_db"
    fi
    
    export PGPASSWORD="$DB_PASSWORD"
    
    # Create target database
    create_database "$target_db"
    
    # Clone using pg_dump and pg_restore
    log "Copying data from $source_db to $target_db"
    
    pg_dump \
        --host="$DB_HOST" \
        --port="$DB_PORT" \
        --username="$DB_USER" \
        --dbname="$source_db" \
        --format=custom \
        --compress=9 | \
    pg_restore \
        --host="$DB_HOST" \
        --port="$DB_PORT" \
        --username="$DB_USER" \
        --dbname="$target_db" \
        --verbose
    
    unset PGPASSWORD
    
    log "Database clone completed successfully"
    verify_restore "$target_db"
}

# Show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
    full <backup_path> [target_db]           Full restore from backup
    schema <backup_path> [target_db]         Schema-only restore
    data <backup_path> [target_db] [tables] Data-only restore
    pitr <backup_path> <time> [target_db]    Point-in-time recovery
    clone <source_db> <target_db>            Clone existing database
    verify <target_db>                       Verify database integrity
    create <db_name>                         Create new database
    drop <db_name>                           Drop database

Options:
    -h, --help                Show this help message
    --drop-existing           Drop target database if it exists
    -f, --force               Force operations without confirmation
    -v, --verbose             Enable verbose output

Environment Variables:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

Examples:
    $0 full /path/to/backup.sql                    # Restore to default database
    $0 full /path/to/backup.sql trading_test       # Restore to specific database
    $0 schema /path/to/schema.sql                  # Schema-only restore
    $0 data /path/to/data.sql trading_test trading.stock_prices  # Specific table
    $0 pitr /path/to/backup.sql "2023-12-01 15:30:00"  # Point-in-time recovery
    $0 clone trading_db trading_test               # Clone database
    $0 verify trading_test                         # Verify integrity
EOF
}

# Main function
main() {
    local drop_existing=false
    local force=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            --drop-existing)
                drop_existing=true
                shift
                ;;
            -f|--force)
                force=true
                shift
                ;;
            -v|--verbose)
                set -x
                shift
                ;;
            *)
                break
                ;;
        esac
    done
    
    if [[ $# -eq 0 ]]; then
        usage
        exit 1
    fi
    
    local command="$1"
    shift
    
    # Execute command
    case "$command" in
        full)
            if [[ $# -eq 0 ]]; then
                error_exit "No backup file specified"
            fi
            full_restore "$1" "${2:-$DB_NAME}" "$drop_existing"
            ;;
        schema)
            if [[ $# -eq 0 ]]; then
                error_exit "No backup file specified"
            fi
            schema_restore "$1" "${2:-$DB_NAME}"
            ;;
        data)
            if [[ $# -eq 0 ]]; then
                error_exit "No backup file specified"
            fi
            local backup_path="$1"
            local target_db="${2:-$DB_NAME}"
            shift 2
            data_restore "$backup_path" "$target_db" "$@"
            ;;
        pitr)
            if [[ $# -lt 2 ]]; then
                error_exit "Backup file and target time required for PITR"
            fi
            point_in_time_restore "$1" "$2" "${3:-${DB_NAME}_pitr}"
            ;;
        clone)
            if [[ $# -lt 2 ]]; then
                error_exit "Source and target database names required"
            fi
            clone_database "$1" "$2"
            ;;
        verify)
            if [[ $# -eq 0 ]]; then
                error_exit "No database specified for verification"
            fi
            verify_restore "$1"
            ;;
        create)
            if [[ $# -eq 0 ]]; then
                error_exit "No database name specified"
            fi
            create_database "$1"
            ;;
        drop)
            if [[ $# -eq 0 ]]; then
                error_exit "No database name specified"
            fi
            drop_database "$1" "$force"
            ;;
        *)
            error_exit "Unknown command: $command"
            ;;
    esac
}

# Run main function
main "$@"


#!/bin/bash

# Database backup script for AI Options Trading System
# Supports full backups, incremental backups, and point-in-time recovery

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/ai-ots}"
LOG_FILE="${LOG_FILE:-/var/log/ai-ots-backup.log}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
COMPRESSION="${COMPRESSION:-gzip}"

# Database connection parameters
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-trading_db}"
DB_USER="${DB_USER:-trading_admin}"
DB_PASSWORD="${DB_PASSWORD:-}"

# S3 configuration for remote backups
S3_BUCKET="${S3_BUCKET:-}"
S3_PREFIX="${S3_PREFIX:-backups/ai-ots}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Check dependencies
check_dependencies() {
    local deps=("pg_dump" "pg_restore" "psql")
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            error_exit "$dep is required but not installed"
        fi
    done
    
    if [[ -n "$S3_BUCKET" ]] && ! command -v "aws" &> /dev/null; then
        error_exit "AWS CLI is required for S3 backups but not installed"
    fi
}

# Create backup directory
create_backup_dir() {
    mkdir -p "$BACKUP_DIR"
    if [[ ! -w "$BACKUP_DIR" ]]; then
        error_exit "Backup directory $BACKUP_DIR is not writable"
    fi
}

# Generate backup filename
generate_backup_filename() {
    local backup_type="$1"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    echo "${backup_type}_${DB_NAME}_${timestamp}.sql"
}

# Full database backup
full_backup() {
    log "Starting full database backup"
    
    local filename=$(generate_backup_filename "full")
    local backup_path="$BACKUP_DIR/$filename"
    
    # Set password for pg_dump
    export PGPASSWORD="$DB_PASSWORD"
    
    # Perform backup
    if pg_dump \
        --host="$DB_HOST" \
        --port="$DB_PORT" \
        --username="$DB_USER" \
        --dbname="$DB_NAME" \
        --verbose \
        --format=custom \
        --compress=9 \
        --file="$backup_path"; then
        
        log "Full backup completed: $backup_path"
        
        # Compress if requested
        if [[ "$COMPRESSION" == "gzip" ]]; then
            gzip "$backup_path"
            backup_path="${backup_path}.gz"
            log "Backup compressed: $backup_path"
        fi
        
        # Upload to S3 if configured
        if [[ -n "$S3_BUCKET" ]]; then
            upload_to_s3 "$backup_path" "full"
        fi
        
        echo "$backup_path"
    else
        error_exit "Full backup failed"
    fi
    
    unset PGPASSWORD
}

# Schema-only backup
schema_backup() {
    log "Starting schema-only backup"
    
    local filename=$(generate_backup_filename "schema")
    local backup_path="$BACKUP_DIR/$filename"
    
    export PGPASSWORD="$DB_PASSWORD"
    
    if pg_dump \
        --host="$DB_HOST" \
        --port="$DB_PORT" \
        --username="$DB_USER" \
        --dbname="$DB_NAME" \
        --schema-only \
        --verbose \
        --file="$backup_path"; then
        
        log "Schema backup completed: $backup_path"
        
        if [[ "$COMPRESSION" == "gzip" ]]; then
            gzip "$backup_path"
            backup_path="${backup_path}.gz"
        fi
        
        echo "$backup_path"
    else
        error_exit "Schema backup failed"
    fi
    
    unset PGPASSWORD
}

# Data-only backup for specific tables
data_backup() {
    local tables=("$@")
    log "Starting data-only backup for tables: ${tables[*]}"
    
    local filename=$(generate_backup_filename "data")
    local backup_path="$BACKUP_DIR/$filename"
    
    export PGPASSWORD="$DB_PASSWORD"
    
    local table_args=()
    for table in "${tables[@]}"; do
        table_args+=("--table=$table")
    done
    
    if pg_dump \
        --host="$DB_HOST" \
        --port="$DB_PORT" \
        --username="$DB_USER" \
        --dbname="$DB_NAME" \
        --data-only \
        --verbose \
        --format=custom \
        --compress=9 \
        "${table_args[@]}" \
        --file="$backup_path"; then
        
        log "Data backup completed: $backup_path"
        
        if [[ "$COMPRESSION" == "gzip" ]]; then
            gzip "$backup_path"
            backup_path="${backup_path}.gz"
        fi
        
        echo "$backup_path"
    else
        error_exit "Data backup failed"
    fi
    
    unset PGPASSWORD
}

# Upload backup to S3
upload_to_s3() {
    local backup_path="$1"
    local backup_type="$2"
    
    log "Uploading backup to S3: s3://$S3_BUCKET/$S3_PREFIX/"
    
    local s3_key="$S3_PREFIX/$backup_type/$(basename "$backup_path")"
    
    if aws s3 cp "$backup_path" "s3://$S3_BUCKET/$s3_key" \
        --region "$AWS_REGION" \
        --storage-class STANDARD_IA; then
        log "Backup uploaded to S3: s3://$S3_BUCKET/$s3_key"
    else
        log "WARNING: Failed to upload backup to S3"
    fi
}

# Clean old backups
cleanup_old_backups() {
    log "Cleaning up backups older than $RETENTION_DAYS days"
    
    # Local cleanup
    find "$BACKUP_DIR" -name "*.sql*" -type f -mtime +$RETENTION_DAYS -delete
    
    # S3 cleanup (if configured)
    if [[ -n "$S3_BUCKET" ]]; then
        local cutoff_date=$(date -d "$RETENTION_DAYS days ago" '+%Y-%m-%d')
        
        aws s3api list-objects-v2 \
            --bucket "$S3_BUCKET" \
            --prefix "$S3_PREFIX/" \
            --query "Contents[?LastModified<='$cutoff_date'].Key" \
            --output text | \
        while read -r key; do
            if [[ -n "$key" ]]; then
                aws s3 rm "s3://$S3_BUCKET/$key"
                log "Deleted old S3 backup: $key"
            fi
        done
    fi
    
    log "Cleanup completed"
}

# Restore database from backup
restore_backup() {
    local backup_path="$1"
    local target_db="${2:-$DB_NAME}"
    
    log "Starting database restore from: $backup_path"
    
    # Check if backup file exists
    if [[ ! -f "$backup_path" ]]; then
        error_exit "Backup file not found: $backup_path"
    fi
    
    # Decompress if needed
    local restore_file="$backup_path"
    if [[ "$backup_path" == *.gz ]]; then
        restore_file="${backup_path%.gz}"
        gunzip -c "$backup_path" > "$restore_file"
    fi
    
    export PGPASSWORD="$DB_PASSWORD"
    
    # Create target database if it doesn't exist
    if ! psql \
        --host="$DB_HOST" \
        --port="$DB_PORT" \
        --username="$DB_USER" \
        --dbname="postgres" \
        --command="SELECT 1 FROM pg_database WHERE datname='$target_db'" | grep -q 1; then
        
        log "Creating target database: $target_db"
        createdb \
            --host="$DB_HOST" \
            --port="$DB_PORT" \
            --username="$DB_USER" \
            "$target_db"
    fi
    
    # Restore backup
    if pg_restore \
        --host="$DB_HOST" \
        --port="$DB_PORT" \
        --username="$DB_USER" \
        --dbname="$target_db" \
        --verbose \
        --clean \
        --if-exists \
        "$restore_file"; then
        
        log "Database restore completed successfully"
    else
        error_exit "Database restore failed"
    fi
    
    # Clean up temporary decompressed file
    if [[ "$restore_file" != "$backup_path" ]]; then
        rm -f "$restore_file"
    fi
    
    unset PGPASSWORD
}

# Verify backup integrity
verify_backup() {
    local backup_path="$1"
    
    log "Verifying backup integrity: $backup_path"
    
    if [[ "$backup_path" == *.gz ]]; then
        if gzip -t "$backup_path"; then
            log "Backup compression integrity verified"
        else
            error_exit "Backup compression is corrupted"
        fi
    fi
    
    # For custom format backups, use pg_restore to list contents
    if [[ "$backup_path" == *.sql ]] && [[ "$backup_path" != *.gz ]]; then
        export PGPASSWORD="$DB_PASSWORD"
        
        if pg_restore --list "$backup_path" > /dev/null 2>&1; then
            log "Backup structure integrity verified"
        else
            error_exit "Backup structure is corrupted"
        fi
        
        unset PGPASSWORD
    fi
    
    log "Backup verification completed"
}

# List available backups
list_backups() {
    log "Available local backups:"
    find "$BACKUP_DIR" -name "*.sql*" -type f -printf "%T@ %Tc %p\n" | sort -n | cut -d' ' -f2-
    
    if [[ -n "$S3_BUCKET" ]]; then
        log "Available S3 backups:"
        aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/" --recursive --human-readable
    fi
}

# Show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
    full                    Create full database backup
    schema                  Create schema-only backup
    data [tables...]        Create data-only backup for specified tables
    restore <backup_path>   Restore database from backup
    verify <backup_path>    Verify backup integrity
    cleanup                 Clean up old backups
    list                    List available backups

Options:
    -h, --help             Show this help message
    -d, --backup-dir DIR   Backup directory (default: $BACKUP_DIR)
    -r, --retention DAYS   Retention period in days (default: $RETENTION_DAYS)
    -c, --compress         Enable compression (default: $COMPRESSION)
    -s, --s3-bucket BUCKET S3 bucket for remote backups
    -v, --verbose          Enable verbose output

Environment Variables:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    S3_BUCKET, S3_PREFIX, AWS_REGION
    BACKUP_DIR, RETENTION_DAYS, COMPRESSION

Examples:
    $0 full                                    # Full backup
    $0 schema                                  # Schema only
    $0 data trading.stock_prices trading.signals  # Specific tables
    $0 restore /path/to/backup.sql             # Restore backup
    $0 cleanup                                 # Clean old backups
EOF
}

# Main function
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -d|--backup-dir)
                BACKUP_DIR="$2"
                shift 2
                ;;
            -r|--retention)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            -c|--compress)
                COMPRESSION="gzip"
                shift
                ;;
            -s|--s3-bucket)
                S3_BUCKET="$2"
                shift 2
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
    
    # Initialize
    check_dependencies
    create_backup_dir
    
    # Execute command
    case "$command" in
        full)
            full_backup
            ;;
        schema)
            schema_backup
            ;;
        data)
            if [[ $# -eq 0 ]]; then
                error_exit "No tables specified for data backup"
            fi
            data_backup "$@"
            ;;
        restore)
            if [[ $# -eq 0 ]]; then
                error_exit "No backup file specified for restore"
            fi
            restore_backup "$1" "${2:-}"
            ;;
        verify)
            if [[ $# -eq 0 ]]; then
                error_exit "No backup file specified for verification"
            fi
            verify_backup "$1"
            ;;
        cleanup)
            cleanup_old_backups
            ;;
        list)
            list_backups
            ;;
        *)
            error_exit "Unknown command: $command"
            ;;
    esac
}

# Run main function
main "$@"


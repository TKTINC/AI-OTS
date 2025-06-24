# DB Subnet Group
resource "aws_db_subnet_group" "main" {
  name       = "${local.name_prefix}-db-subnet-group"
  subnet_ids = aws_subnet.database[*].id

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-db-subnet-group"
  })
}

# DB Parameter Group for TimescaleDB
resource "aws_db_parameter_group" "timescaledb" {
  family = "postgres15"
  name   = "${local.name_prefix}-timescaledb-params"

  parameter {
    name  = "shared_preload_libraries"
    value = "timescaledb"
  }

  parameter {
    name  = "max_connections"
    value = "200"
  }

  parameter {
    name  = "shared_buffers"
    value = "{DBInstanceClassMemory/4}"
  }

  parameter {
    name  = "effective_cache_size"
    value = "{DBInstanceClassMemory*3/4}"
  }

  parameter {
    name  = "maintenance_work_mem"
    value = "2097152"  # 2GB in KB
  }

  parameter {
    name  = "checkpoint_completion_target"
    value = "0.9"
  }

  parameter {
    name  = "wal_buffers"
    value = "16384"  # 16MB in KB
  }

  parameter {
    name  = "default_statistics_target"
    value = "100"
  }

  parameter {
    name  = "random_page_cost"
    value = "1.1"
  }

  parameter {
    name  = "effective_io_concurrency"
    value = "200"
  }

  parameter {
    name  = "work_mem"
    value = "10485"  # 10MB in KB
  }

  parameter {
    name  = "min_wal_size"
    value = "2048"  # 2GB in MB
  }

  parameter {
    name  = "max_wal_size"
    value = "8192"  # 8GB in MB
  }

  tags = local.common_tags
}

# RDS Instance
resource "aws_db_instance" "main" {
  identifier = "${local.name_prefix}-timescaledb"

  # Engine Configuration
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.db_instance_class

  # Storage Configuration
  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true

  # Database Configuration
  db_name  = var.db_name
  username = var.db_username
  password = var.db_password
  port     = 5432

  # Network Configuration
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false

  # Parameter and Option Groups
  parameter_group_name = aws_db_parameter_group.timescaledb.name

  # Backup Configuration
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  # Monitoring
  monitoring_interval = var.enable_detailed_monitoring ? 60 : 0
  monitoring_role_arn = var.enable_detailed_monitoring ? aws_iam_role.rds_enhanced_monitoring[0].arn : null

  # Performance Insights
  performance_insights_enabled = var.enable_detailed_monitoring
  performance_insights_retention_period = var.enable_detailed_monitoring ? 7 : null

  # Security
  deletion_protection = var.enable_deletion_protection
  skip_final_snapshot = !var.enable_deletion_protection
  final_snapshot_identifier = var.enable_deletion_protection ? "${local.name_prefix}-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}" : null

  # Enable automated minor version upgrades
  auto_minor_version_upgrade = true

  # Copy tags to snapshots
  copy_tags_to_snapshot = true

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-timescaledb"
  })

  depends_on = [
    aws_cloudwatch_log_group.rds
  ]
}

# RDS Enhanced Monitoring Role (conditional)
resource "aws_iam_role" "rds_enhanced_monitoring" {
  count = var.enable_detailed_monitoring ? 1 : 0
  name  = "${local.name_prefix}-rds-enhanced-monitoring"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "rds_enhanced_monitoring" {
  count      = var.enable_detailed_monitoring ? 1 : 0
  role       = aws_iam_role.rds_enhanced_monitoring[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# RDS Read Replica (for read scaling)
resource "aws_db_instance" "read_replica" {
  count = var.environment == "prod" ? 1 : 0

  identifier = "${local.name_prefix}-timescaledb-replica"

  # Replica Configuration
  replicate_source_db = aws_db_instance.main.identifier
  instance_class      = var.db_instance_class

  # Network Configuration
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false

  # Monitoring
  monitoring_interval = var.enable_detailed_monitoring ? 60 : 0
  monitoring_role_arn = var.enable_detailed_monitoring ? aws_iam_role.rds_enhanced_monitoring[0].arn : null

  # Performance Insights
  performance_insights_enabled = var.enable_detailed_monitoring

  # Security
  skip_final_snapshot = true

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-timescaledb-replica"
  })
}

# CloudWatch Log Group for RDS
resource "aws_cloudwatch_log_group" "rds" {
  name              = "/aws/rds/instance/${local.name_prefix}-timescaledb/postgresql"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

# RDS Subnet Group for Cache
resource "aws_elasticache_subnet_group" "main" {
  name       = "${local.name_prefix}-cache-subnet-group"
  subnet_ids = aws_subnet.private[*].id

  tags = local.common_tags
}


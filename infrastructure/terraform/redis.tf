# ElastiCache Parameter Group
resource "aws_elasticache_parameter_group" "redis" {
  family = "redis7.x"
  name   = "${local.name_prefix}-redis-params"

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }

  parameter {
    name  = "timeout"
    value = "300"
  }

  parameter {
    name  = "tcp-keepalive"
    value = "300"
  }

  parameter {
    name  = "maxclients"
    value = "10000"
  }

  tags = local.common_tags
}

# ElastiCache Redis Cluster
resource "aws_elasticache_cluster" "main" {
  cluster_id           = "${local.name_prefix}-redis"
  engine               = "redis"
  node_type            = var.redis_node_type
  num_cache_nodes      = var.redis_num_cache_nodes
  parameter_group_name = aws_elasticache_parameter_group.redis.name
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.redis.id]

  # Enable automatic failover for multi-AZ
  az_mode = var.redis_num_cache_nodes > 1 ? "cross-az" : "single-az"

  # Maintenance window
  maintenance_window = "sun:05:00-sun:06:00"

  # Snapshot configuration
  snapshot_retention_limit = 5
  snapshot_window         = "03:00-05:00"

  # Enable at-rest encryption
  at_rest_encryption_enabled = true

  # Enable in-transit encryption
  transit_encryption_enabled = true

  # Auth token for security (optional, can be enabled later)
  # auth_token = var.redis_auth_token

  # Notification topic for events
  notification_topic_arn = aws_sns_topic.alerts.arn

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-redis"
  })
}

# ElastiCache Replication Group (for high availability in production)
resource "aws_elasticache_replication_group" "main" {
  count = var.environment == "prod" ? 1 : 0

  replication_group_id       = "${local.name_prefix}-redis-cluster"
  description                = "Redis cluster for ${local.name_prefix}"
  
  # Node configuration
  node_type                  = var.redis_node_type
  port                       = 6379
  parameter_group_name       = aws_elasticache_parameter_group.redis.name
  
  # Cluster configuration
  num_cache_clusters         = 2
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  # Network configuration
  subnet_group_name = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]

  # Maintenance and backup
  maintenance_window       = "sun:05:00-sun:06:00"
  snapshot_retention_limit = 5
  snapshot_window         = "03:00-05:00"

  # Security
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  # Logging
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis.name
    destination_type = "cloudwatch-logs"
    log_format       = "text"
    log_type         = "slow-log"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-redis-cluster"
  })
}

# CloudWatch Log Group for Redis
resource "aws_cloudwatch_log_group" "redis" {
  name              = "/aws/elasticache/redis/${local.name_prefix}"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

# SNS Topic for Redis Alerts
resource "aws_sns_topic" "alerts" {
  name = "${local.name_prefix}-alerts"

  tags = local.common_tags
}

# CloudWatch Alarms for Redis
resource "aws_cloudwatch_metric_alarm" "redis_cpu" {
  alarm_name          = "${local.name_prefix}-redis-cpu-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ElastiCache"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors redis cpu utilization"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    CacheClusterId = aws_elasticache_cluster.main.cluster_id
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "redis_memory" {
  alarm_name          = "${local.name_prefix}-redis-memory-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "DatabaseMemoryUsagePercentage"
  namespace           = "AWS/ElastiCache"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors redis memory utilization"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    CacheClusterId = aws_elasticache_cluster.main.cluster_id
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "redis_connections" {
  alarm_name          = "${local.name_prefix}-redis-connections"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CurrConnections"
  namespace           = "AWS/ElastiCache"
  period              = "300"
  statistic           = "Average"
  threshold           = "100"
  alarm_description   = "This metric monitors redis connection count"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    CacheClusterId = aws_elasticache_cluster.main.cluster_id
  }

  tags = local.common_tags
}

# Redis Configuration for Application
locals {
  redis_config = {
    host     = aws_elasticache_cluster.main.cache_nodes[0].address
    port     = aws_elasticache_cluster.main.cache_nodes[0].port
    url      = "redis://${aws_elasticache_cluster.main.cache_nodes[0].address}:${aws_elasticache_cluster.main.cache_nodes[0].port}/0"
    ssl      = true
    timeout  = 5
    max_connections = 100
    
    # Cache TTL settings (in seconds)
    ttl = {
      stock_prices    = 300    # 5 minutes
      options_data    = 300    # 5 minutes
      signals         = 1800   # 30 minutes
      user_sessions   = 3600   # 1 hour
      rate_limits     = 3600   # 1 hour
    }
  }
}


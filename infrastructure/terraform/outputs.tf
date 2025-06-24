# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

# Database Outputs
output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}

output "rds_port" {
  description = "RDS instance port"
  value       = aws_db_instance.main.port
}

output "database_url" {
  description = "Database connection URL"
  value       = "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.main.endpoint}:${aws_db_instance.main.port}/${var.db_name}"
  sensitive   = true
}

# Redis Outputs
output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_cluster.main.cache_nodes[0].address
  sensitive   = true
}

output "redis_port" {
  description = "Redis cluster port"
  value       = aws_elasticache_cluster.main.cache_nodes[0].port
}

output "redis_url" {
  description = "Redis connection URL"
  value       = "redis://${aws_elasticache_cluster.main.cache_nodes[0].address}:${aws_elasticache_cluster.main.cache_nodes[0].port}/0"
  sensitive   = true
}

# ECS Outputs
output "ecs_cluster_id" {
  description = "ECS cluster ID"
  value       = aws_ecs_cluster.main.id
}

output "ecs_cluster_arn" {
  description = "ECS cluster ARN"
  value       = aws_ecs_cluster.main.arn
}

output "ecs_service_security_group_id" {
  description = "Security group ID for ECS services"
  value       = aws_security_group.ecs_service.id
}

# Load Balancer Outputs
output "alb_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the load balancer"
  value       = aws_lb.main.zone_id
}

output "alb_arn" {
  description = "ARN of the load balancer"
  value       = aws_lb.main.arn
}

# IAM Outputs
output "ecs_task_execution_role_arn" {
  description = "ARN of the ECS task execution role"
  value       = aws_iam_role.ecs_task_execution.arn
}

output "ecs_task_role_arn" {
  description = "ARN of the ECS task role"
  value       = aws_iam_role.ecs_task.arn
}

# S3 Outputs
output "s3_bucket_logs" {
  description = "S3 bucket for logs"
  value       = aws_s3_bucket.logs.bucket
}

output "s3_bucket_data" {
  description = "S3 bucket for data"
  value       = aws_s3_bucket.data.bucket
}

# CloudWatch Outputs
output "cloudwatch_log_group_arn" {
  description = "ARN of the main CloudWatch log group"
  value       = aws_cloudwatch_log_group.main.arn
}

# Environment Variables for Applications
output "environment_variables" {
  description = "Environment variables for applications"
  value = {
    DATABASE_URL     = "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.main.endpoint}:${aws_db_instance.main.port}/${var.db_name}"
    REDIS_URL        = "redis://${aws_elasticache_cluster.main.cache_nodes[0].address}:${aws_elasticache_cluster.main.cache_nodes[0].port}/0"
    AWS_REGION       = var.aws_region
    ENVIRONMENT      = var.environment
    LOG_GROUP        = aws_cloudwatch_log_group.main.name
    S3_BUCKET_LOGS   = aws_s3_bucket.logs.bucket
    S3_BUCKET_DATA   = aws_s3_bucket.data.bucket
    DATABENTO_API_KEY = var.databento_api_key
    DATABENTO_DATASET = var.databento_dataset
    DATABENTO_SYMBOLS = var.databento_symbols
  }
  sensitive = true
}


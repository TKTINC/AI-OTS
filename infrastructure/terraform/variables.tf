# Project Configuration
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "ai-ots"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}

# Database Configuration
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "db_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 100
}

variable "db_max_allocated_storage" {
  description = "RDS maximum allocated storage in GB"
  type        = number
  default     = 1000
}

variable "db_username" {
  description = "Database master username"
  type        = string
  default     = "trading_admin"
  sensitive   = true
}

variable "db_password" {
  description = "Database master password"
  type        = string
  default     = "PLACEHOLDER_DB_PASSWORD"
  sensitive   = true
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "trading_db"
}

# Redis Configuration
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.medium"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes"
  type        = number
  default     = 1
}

variable "redis_parameter_group_name" {
  description = "Redis parameter group name"
  type        = string
  default     = "default.redis7"
}

# ECS Configuration
variable "ecs_task_cpu" {
  description = "CPU units for ECS tasks"
  type        = number
  default     = 512
}

variable "ecs_task_memory" {
  description = "Memory for ECS tasks"
  type        = number
  default     = 1024
}

variable "ecs_desired_count" {
  description = "Desired number of ECS tasks"
  type        = number
  default     = 2
}

# Application Configuration
variable "databento_api_key" {
  description = "Databento API key"
  type        = string
  default     = "PLACEHOLDER_DATABENTO_API_KEY"
  sensitive   = true
}

variable "databento_dataset" {
  description = "Databento dataset"
  type        = string
  default     = "XNAS.ITCH,OPRA.PILLAR"
}

variable "databento_symbols" {
  description = "Trading symbols to track"
  type        = string
  default     = "AAPL,GOOGL,AMZN,MSFT,NVDA,META,AVGO,SPY,QQQ"
}

# Monitoring Configuration
variable "enable_detailed_monitoring" {
  description = "Enable detailed CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

# Security Configuration
variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the application"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict this in production
}

variable "enable_deletion_protection" {
  description = "Enable deletion protection for critical resources"
  type        = bool
  default     = false  # Set to true in production
}


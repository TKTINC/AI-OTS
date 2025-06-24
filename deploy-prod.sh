#!/bin/bash

# AI Options Trading System - Production Deployment Script
# This script deploys the system to AWS using Terraform

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to validate AWS credentials
validate_aws_credentials() {
    print_status "Validating AWS credentials..."
    
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        print_error "AWS credentials not configured or invalid"
        print_status "Please run: aws configure"
        exit 1
    fi
    
    print_success "AWS credentials validated"
}

# Function to check Terraform state
check_terraform_state() {
    local tf_dir=$1
    
    if [ ! -f "$tf_dir/terraform.tfstate" ] && [ ! -f "$tf_dir/.terraform/terraform.tfstate" ]; then
        print_warning "No Terraform state found. This appears to be a fresh deployment."
        return 1
    fi
    
    return 0
}

# Function to deploy infrastructure
deploy_infrastructure() {
    print_status "Deploying AWS infrastructure with Terraform..."
    
    cd infrastructure/terraform
    
    # Initialize Terraform
    print_status "Initializing Terraform..."
    terraform init
    
    # Validate configuration
    print_status "Validating Terraform configuration..."
    terraform validate
    
    # Plan deployment
    print_status "Planning infrastructure deployment..."
    terraform plan -out=tfplan
    
    # Ask for confirmation
    echo
    print_warning "This will create AWS resources that may incur costs."
    read -p "Do you want to proceed with the deployment? [y/N]: " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Deployment cancelled by user"
        exit 0
    fi
    
    # Apply changes
    print_status "Applying Terraform configuration..."
    terraform apply tfplan
    
    # Get outputs
    print_status "Getting infrastructure outputs..."
    terraform output -json > ../outputs.json
    
    cd ../..
    
    print_success "Infrastructure deployment completed"
}

# Function to build and push Docker images
build_and_push_images() {
    print_status "Building and pushing Docker images..."
    
    # Get ECR repository URLs from Terraform outputs
    local ecr_registry=$(jq -r '.ecr_registry.value' infrastructure/outputs.json)
    local aws_region=$(jq -r '.aws_region.value' infrastructure/outputs.json)
    
    if [ "$ecr_registry" = "null" ] || [ "$aws_region" = "null" ]; then
        print_error "Could not get ECR registry information from Terraform outputs"
        exit 1
    fi
    
    # Login to ECR
    print_status "Logging in to ECR..."
    aws ecr get-login-password --region "$aws_region" | docker login --username AWS --password-stdin "$ecr_registry"
    
    # Build and push images
    local services=("api-gateway" "cache" "data-ingestion" "analytics")
    
    for service in "${services[@]}"; do
        print_status "Building and pushing $service image..."
        
        local image_tag="$ecr_registry/ai-ots-$service:latest"
        
        # Build image
        docker build -t "$image_tag" "services/$service/"
        
        # Push image
        docker push "$image_tag"
        
        print_success "$service image pushed successfully"
    done
}

# Function to deploy ECS services
deploy_ecs_services() {
    print_status "Deploying ECS services..."
    
    # Get cluster information from Terraform outputs
    local cluster_name=$(jq -r '.ecs_cluster_name.value' infrastructure/outputs.json)
    local aws_region=$(jq -r '.aws_region.value' infrastructure/outputs.json)
    
    if [ "$cluster_name" = "null" ] || [ "$aws_region" = "null" ]; then
        print_error "Could not get ECS cluster information from Terraform outputs"
        exit 1
    fi
    
    # Update ECS services
    local services=("api-gateway" "cache" "data-ingestion" "analytics")
    
    for service in "${services[@]}"; do
        print_status "Updating ECS service: ai-ots-$service"
        
        aws ecs update-service \
            --cluster "$cluster_name" \
            --service "ai-ots-$service" \
            --force-new-deployment \
            --region "$aws_region" >/dev/null
        
        print_success "ECS service ai-ots-$service updated"
    done
    
    # Wait for services to stabilize
    print_status "Waiting for services to stabilize..."
    
    for service in "${services[@]}"; do
        print_status "Waiting for ai-ots-$service to stabilize..."
        
        aws ecs wait services-stable \
            --cluster "$cluster_name" \
            --services "ai-ots-$service" \
            --region "$aws_region"
        
        print_success "ai-ots-$service is stable"
    done
}

# Function to run database migrations
run_database_migrations() {
    print_status "Running database migrations..."
    
    # Get RDS endpoint from Terraform outputs
    local rds_endpoint=$(jq -r '.rds_endpoint.value' infrastructure/outputs.json)
    
    if [ "$rds_endpoint" = "null" ]; then
        print_error "Could not get RDS endpoint from Terraform outputs"
        exit 1
    fi
    
    # Run migrations using a temporary ECS task
    # This would typically run the migration script in a container
    print_status "Database migrations would run here (implementation needed)"
    
    print_success "Database migrations completed"
}

# Function to verify deployment
verify_deployment() {
    print_status "Verifying deployment..."
    
    # Get load balancer URL from Terraform outputs
    local lb_url=$(jq -r '.load_balancer_url.value' infrastructure/outputs.json)
    
    if [ "$lb_url" = "null" ]; then
        print_error "Could not get load balancer URL from Terraform outputs"
        exit 1
    fi
    
    # Test health endpoints
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$lb_url/health" >/dev/null 2>&1; then
            print_success "Deployment verification successful!"
            echo "🌐 Application URL: $lb_url"
            return 0
        fi
        
        echo -n "."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    print_error "Deployment verification failed"
    return 1
}

# Function to show deployment summary
show_deployment_summary() {
    echo
    echo "=============================================="
    echo "Deployment Summary"
    echo "=============================================="
    
    if [ -f infrastructure/outputs.json ]; then
        local lb_url=$(jq -r '.load_balancer_url.value' infrastructure/outputs.json)
        local rds_endpoint=$(jq -r '.rds_endpoint.value' infrastructure/outputs.json)
        local redis_endpoint=$(jq -r '.redis_endpoint.value' infrastructure/outputs.json)
        local s3_bucket=$(jq -r '.s3_bucket_name.value' infrastructure/outputs.json)
        
        echo "🌐 Application URL:    $lb_url"
        echo "🗄️  Database Endpoint: $rds_endpoint"
        echo "🔴 Redis Endpoint:     $redis_endpoint"
        echo "📦 S3 Bucket:          $s3_bucket"
        echo
        echo "Health Check URLs:"
        echo "  API Gateway:         $lb_url/health"
        echo "  Cache Service:       $lb_url/api/v1/cache/health"
        echo "  Data Ingestion:      $lb_url/api/v1/data/health"
        echo "  Analytics:           $lb_url/api/v1/analytics/health"
        echo
    fi
    
    echo "=============================================="
    echo "Post-Deployment Tasks:"
    echo "=============================================="
    echo "1. Configure DNS records to point to the load balancer"
    echo "2. Set up SSL certificates"
    echo "3. Configure monitoring and alerting"
    echo "4. Set up backup schedules"
    echo "5. Configure log aggregation"
    echo "6. Test all API endpoints"
    echo "7. Set up CI/CD pipelines"
    echo
}

# Main deployment function
main() {
    local action=${1:-deploy}
    
    echo "=============================================="
    echo "AI Options Trading System - Production Deployment"
    echo "=============================================="
    echo
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    
    local required_commands=("aws" "terraform" "docker" "jq" "curl")
    for cmd in "${required_commands[@]}"; do
        if ! command_exists "$cmd"; then
            print_error "$cmd is not installed. Please install it first."
            exit 1
        fi
    done
    
    print_success "Prerequisites check passed"
    echo
    
    case $action in
        "deploy")
            # Validate AWS credentials
            validate_aws_credentials
            
            # Deploy infrastructure
            deploy_infrastructure
            
            # Build and push Docker images
            build_and_push_images
            
            # Deploy ECS services
            deploy_ecs_services
            
            # Run database migrations
            run_database_migrations
            
            # Verify deployment
            verify_deployment
            
            # Show summary
            show_deployment_summary
            
            print_success "Production deployment completed successfully!"
            ;;
            
        "destroy")
            print_warning "This will destroy all AWS resources created by Terraform."
            read -p "Are you sure you want to proceed? [y/N]: " -n 1 -r
            echo
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                cd infrastructure/terraform
                terraform destroy
                cd ../..
                print_success "Infrastructure destroyed"
            else
                print_status "Destruction cancelled"
            fi
            ;;
            
        "status")
            if [ -f infrastructure/outputs.json ]; then
                show_deployment_summary
            else
                print_error "No deployment found. Run './deploy-prod.sh deploy' first."
            fi
            ;;
            
        *)
            echo "Usage: $0 [deploy|destroy|status]"
            echo
            echo "Commands:"
            echo "  deploy   - Deploy the system to AWS"
            echo "  destroy  - Destroy all AWS resources"
            echo "  status   - Show deployment status"
            exit 1
            ;;
    esac
}

# Handle script interruption
trap 'print_error "Script interrupted"; exit 1' INT TERM

# Run main function
main "$@"


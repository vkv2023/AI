==============================================================
         Production CI/CD Architecture
==============================================================
Developer Push Code
        ↓
GitHub 
        ↓
GitHub Actions (CI)
        ↓
Build Docker Image using Github commit ID
        ↓
Push Image → Amazon ECR
        ↓
Security & Vulnerability Scan (SonarQube, Trivy, Snyk, AWS Inspector)
        ↓
Deploy → ECS / EKS
        ↓
Load Balancer Health Check
        ↓
Traffic switched to new version
        ↓
Add Multi-region and AutoRollback functionality


.github
 └── workflows
     ├── test.yml (run tests)
     ├── build-image.yml 
     ├── build-ecr.yml (Push to ECR and download from ECR for scanning)
     ├── security-scan.yml Scan the image using trivy)
     ├── deploy-ecs.yml (Deploy on container)
     ├── health-check.yml (Health Check beofre prodcution live)
     └── production-live.yml (Add Canary or Blue/Green release)
     └── Multi-region release and Auto Rollback

GitHub Actions
     ↓
OIDC Authentication (short-lived, no manual tokens needed, automatic rotation, safer)
     ↓
Assume IAM Role
     ↓
Push / Pull Image from ECR
     ↓
Trivy Security Scan
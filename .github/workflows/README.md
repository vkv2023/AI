Git Push
   ↓
Workflow 1
Build Docker Image
   ↓
Tag with Commit SHA
   ↓
Push → ECR
   ↓
Workflow 2 starts
   ↓
ECS pulls new image
   ↓
New version deployed

==============================================================
         Production CI/CD Architecture
==============================================================

Developer Push Code
        ↓
GitHub
        ↓
GitHub Actions (CI)
        ↓
Build Docker Image
        ↓
Security & Vulnerability Scan (SonarQube, Trivy, Snyk, AWS Inspector)
        ↓
Push Image → Amazon ECR
        ↓
Deploy → ECS / EKS
        ↓
Load Balancer Health Check
        ↓
Traffic switched to new version 


.github
 └── workflows
     ├── test.yml
     ├── build-image.yml
     ├── security-scan.yml
     ├── build-ecr.yml
     ├── deploy-ecs.yml
     ├── health-check.yml
     └── production-live.yml
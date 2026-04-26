# EC2 + RDS + S3 Deployment

This is the low-cost AWS path for LIBRA before ECS/ALB/NAT are justified.

## Target Shape

```text
EC2
  Docker Compose
    caddy
    libra-agent
    libra-backend later

RDS MySQL
S3
Claude API
```

Do not create a NAT Gateway or ALB for the MVP. They add fixed cost and are not needed for one EC2 host.

## AWS Resources

Create these manually first:

- EC2: Ubuntu, `t4g.micro` or `t3.micro`
- Security group:
  - inbound `22` from your IP
  - inbound `80` from `0.0.0.0/0`
  - inbound `443` from `0.0.0.0/0`
- RDS MySQL:
  - Single-AZ
  - `db.t4g.micro` or `db.t3.micro`
  - 20 GB storage
  - storage autoscaling off
  - Multi-AZ off
  - Performance Insights off
  - Enhanced Monitoring off
- S3 bucket for LIBRA documents and collected artifacts
- AWS Budget alerts at 30000, 40000, and 50000 KRW equivalent

## EC2 Bootstrap

On the EC2 host:

```bash
curl -fsSL https://raw.githubusercontent.com/YOUR_ORG/YOUR_REPO/main/scripts/ec2-bootstrap.sh -o ec2-bootstrap.sh
bash ec2-bootstrap.sh
```

Log out and back in after Docker is installed.

## First Manual Deploy

```bash
git clone https://github.com/YOUR_ORG/YOUR_REPO.git libra-agent
cd libra-agent
cp .env.prod.example .env.prod
vi .env.prod
```

Required values:

```text
ANTHROPIC_API_KEY=
LIBRA_DB_HOST=
LIBRA_DB_USER=
LIBRA_DB_PASSWORD=
LIBRA_S3_BUCKET=
```

Start only the agent and Caddy:

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build agent caddy
```

Check:

```bash
curl http://localhost/health
docker compose --env-file .env.prod -f docker-compose.prod.yml logs -f agent
```

## With Spring Boot API Later

When `libra-backend` exists and its image is available:

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml --profile api up -d
```

Then set:

```text
LIBRA_UPSTREAM=api:8080
```

and restart Caddy.

## CI/CD Direction

Two workflows are included:

- `.github/workflows/ci.yml`
- `.github/workflows/deploy-ec2.yml`

Add these GitHub repository secrets:

```text
EC2_HOST
EC2_USER
EC2_SSH_KEY
EC2_PORT       optional, defaults to 22
EC2_APP_DIR    optional, defaults to /opt/libra/app/libra-agent
```

The first deployment should create the app directory manually:

```bash
sudo mkdir -p /opt/libra/app
sudo chown -R "$USER:$USER" /opt/libra
cd /opt/libra/app
git clone https://github.com/YOUR_ORG/YOUR_REPO.git libra-agent
cd libra-agent
cp .env.prod.example .env.prod
vi .env.prod
```

After that, GitHub Actions can:

1. Run tests.
2. Build Docker image.
3. SSH into EC2.
4. Pull the latest Git commit.
5. Rebuild and restart Docker Compose.

This first version intentionally avoids ECR to keep the deployment easy. If build time becomes painful, add ECR later and switch the deploy step to `docker compose pull && docker compose up -d`.

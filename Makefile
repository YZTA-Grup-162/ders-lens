# ==== Development ====
.PHONY: dev dev-logs
dev:
	docker compose -f docker-compose.dev.yml up --build

dev-logs:
	docker compose -f docker-compose.dev.yml logs -f

# ==== Production (local) ====
.PHONY: prod prod-logs
prod:
	docker compose -f docker-compose-final.yml up --build -d

prod-logs:
	docker compose -f docker-compose-final.yml logs -f

# ==== Fly.io Deployment ====
.PHONY: fly-init fly-secrets fly-deploy fly-status
fly-init:
	flyctl auth login
	flyctl launch --name ders-lens --region ord --dockerfile ai-service/Dockerfile --no-deploy

fly-secrets:
	flyctl secrets set \
	  BASIC_AUTH_USERNAME=admin \
	  BASIC_AUTH_PASSWORD=SuperSecret123 \
	  DATABASE_URL=postgresql://postgres:password@db:5432/ders_lens \
	  AI_SERVICE_URL=https://ders-lens.fly.dev

fly-deploy:
	flyctl deploy

fly-status:
	flyctl status

# ==== Helpers ====
.PHONY: clean
clean:
	docker compose -f docker-compose.dev.yml down --volumes
	docker compose -f docker-compose-final.yml down --volumes

#!/usr/bin/env bash
set -euo pipefail

LIBRA_USER="${LIBRA_USER:-libra}"
LIBRA_HOME="${LIBRA_HOME:-/opt/libra}"
AGENT_ROOT="${AGENT_ROOT:-${LIBRA_HOME}/agent}"
KNOWLEDGE_CACHE_DIR="${KNOWLEDGE_CACHE_DIR:-${LIBRA_HOME}/knowledge/current}"
AGENT_ENV_FILE="${AGENT_ENV_FILE:-/etc/libra/agent.env}"
AGENT_SERVICE_FILE="/etc/systemd/system/libra-agent.service"
BOOTSTRAP_USER="${SUDO_USER:-$USER}"

sudo apt-get update
sudo apt-get install -y ca-certificates curl git gnupg software-properties-common awscli

if ! apt-cache show python3.12 >/dev/null 2>&1; then
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
fi

sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
python3.12 -m venv --help >/dev/null

sudo install -m 0755 -d /etc/apt/keyrings
if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
fi
sudo chmod a+r /etc/apt/keyrings/docker.gpg

. /etc/os-release
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

if [ "${BOOTSTRAP_USER}" != "root" ]; then
  sudo usermod -aG docker "${BOOTSTRAP_USER}"
fi

if ! id -u "${LIBRA_USER}" >/dev/null 2>&1; then
  sudo useradd --system --home-dir "${LIBRA_HOME}" --shell /usr/sbin/nologin "${LIBRA_USER}"
fi

sudo install -d -o "${LIBRA_USER}" -g "${LIBRA_USER}" -m 0750 \
  "${LIBRA_HOME}" \
  "${AGENT_ROOT}" \
  "${AGENT_ROOT}/app" \
  "${KNOWLEDGE_CACHE_DIR}" \
  "${LIBRA_HOME}/data/agent-outputs" \
  "${LIBRA_HOME}/data/knowledge/agent_refresh"
sudo install -d -m 0755 \
  "${LIBRA_HOME}/data/caddy-data" \
  "${LIBRA_HOME}/data/caddy-config"
sudo chown -R "${LIBRA_USER}:${LIBRA_USER}" \
  "${AGENT_ROOT}" \
  "${LIBRA_HOME}/knowledge" \
  "${LIBRA_HOME}/data/agent-outputs" \
  "${LIBRA_HOME}/data/knowledge"

sudo install -d -o root -g "${LIBRA_USER}" -m 0750 /etc/libra
if [ ! -f "${AGENT_ENV_FILE}" ]; then
  sudo tee "${AGENT_ENV_FILE}" > /dev/null <<'EOF'
# Fill from .env.prod.example. Do not commit real values.
# Required before deploy starts libra-agent:
# DATABASE_URL=postgresql://...
# ANTHROPIC_API_KEY=...
# LIBRA_DOMAIN_AGENTS_ENABLED=true
EOF
fi
sudo chown root:"${LIBRA_USER}" "${AGENT_ENV_FILE}"
sudo chmod 0640 "${AGENT_ENV_FILE}"

sudo tee "${AGENT_SERVICE_FILE}" > /dev/null <<EOF
[Unit]
Description=Libra Agent FastAPI service
After=network-online.target
Wants=network-online.target

[Service]
User=${LIBRA_USER}
Group=${LIBRA_USER}
WorkingDirectory=${AGENT_ROOT}/app
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=${AGENT_ENV_FILE}
ExecStart=${AGENT_ROOT}/.venv/bin/libra-agent-api
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload

echo "Docker is installed. Log out and back in once, then run docker compose commands without sudo."
echo "Python 3.12 is installed for the libra-agent deploy venv."
echo "AWS CLI is installed for SSM deploy artifact downloads."
echo "Prepared ${LIBRA_USER} user, ${AGENT_ROOT}, ${KNOWLEDGE_CACHE_DIR}, ${AGENT_ENV_FILE}, and libra-agent.service."
echo "Fill ${AGENT_ENV_FILE} from .env.prod.example before running the GitHub Actions deploy."

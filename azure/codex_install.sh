sudo apt update
sudo apt install -y curl
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo bash -
sudo apt-get install -y nodejs

sudo npm install -g @openai/codex
codex --sandbox workspace-write --config sandbox_workspace_write.network_access=true
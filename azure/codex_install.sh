sudo apt update
sudo apt install -y curl
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo bash -
sudo apt-get install -y nodejs

sudo npm install -g @openai/codex
codex --sandbox workspace-write --config sandbox_workspace_write.network_access=true


# config.toml
# [mcp_servers.context7]
# args = ["-y", "@upstash/context7-mcp", "--api-key", "ctx7sk-0478d348-d670-439f-a4a4-3bb073445100"]
# command = "npx"
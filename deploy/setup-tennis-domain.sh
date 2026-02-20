#!/bin/bash
# Setup tennis.playfullife.com on Hetzner
# Run this on the Hetzner server after adding DNS record

set -e

DOMAIN="tennis.playfullife.com"
NGINX_CONF="/etc/nginx/sites-available/$DOMAIN"

echo "Setting up $DOMAIN..."

# 1. Install nginx if not present
if ! command -v nginx &> /dev/null; then
    echo "Installing nginx..."
    apt update && apt install -y nginx
fi

# 2. Install certbot if not present
if ! command -v certbot &> /dev/null; then
    echo "Installing certbot..."
    apt install -y certbot python3-certbot-nginx
fi

# 3. Create nginx config
cat > "$NGINX_CONF" << 'EOF'
server {
    listen 80;
    server_name tennis.playfullife.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
EOF

# 4. Enable site
ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/

# 5. Test and reload nginx
nginx -t && systemctl reload nginx

echo "Nginx configured. Now obtaining SSL certificate..."

# 6. Get SSL certificate
certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m your-email@example.com

echo ""
echo "Setup complete!"
echo "  Video gallery: https://$DOMAIN/"
echo "  Dashboard:     https://$DOMAIN/dash"
echo ""

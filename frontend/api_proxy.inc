# Shared proxy settings for the API. Included by both /api/ and /api/token,
# which differ only in which rate-limit zone they use — keeping the proxy
# directives in one place means the two cannot drift apart.

# Service name from docker-compose (the plan calls this "backend-api"; the
# compose service is `api`).
proxy_pass http://api:8000/;

proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;

# Generation on CPU regularly exceeds Nginx's 60s default, and a truncated
# answer looks like a backend crash from the UI. The VM has 2 cores, so a
# verified /query there is slower than the ~15s measured on the host.
proxy_read_timeout 300s;
proxy_send_timeout 300s;

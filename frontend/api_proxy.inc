# Shared proxy settings for the API. Included by both /api/ and /api/token.
#
# proxy_pass is deliberately NOT here — each location must set its own.
# Nginx replaces "the part of the request URI matching the location" with the
# proxy_pass URI, so a shared `proxy_pass http://api:8000/` means:
#
#   location /api/        ->  /api/query  becomes  /query    correct
#   location = /api/token ->  /api/token  becomes  /         404
#
# The exact-match location consumes the whole path, so it needs the full
# upstream path spelled out. Sharing one proxy_pass silently broke login.

proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;

# Generation on CPU regularly exceeds Nginx's 60s default, and a truncated
# answer looks like a backend crash from the UI. The VM has 2 cores, so a
# verified /query there is slower than the ~15s measured on the host.
proxy_read_timeout 300s;
proxy_send_timeout 300s;

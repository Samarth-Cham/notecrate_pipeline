# Plan section 3.3: "Custom Dockerfile (React build -> Nginx)".
# Multi-stage so the node toolchain never reaches the runtime image — the
# result is static assets plus Nginx, a few tens of MB.

FROM node:22-alpine AS build
WORKDIR /app

# Copy manifests first so `npm ci` is cached until dependencies actually change.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:1.27-alpine AS runtime
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

# Compose healthchecks need something to poll that doesn't depend on the
# backend being up.
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://localhost/ >/dev/null 2>&1 || exit 1

# Common base
FROM node:20-slim AS base
WORKDIR /app
RUN npm install -g bun
COPY package*.json bun.lockb* ./
RUN bun install
COPY . .

# Dev stage – vite dev server on port 80
FROM base AS dev
ENV PORT=80
EXPOSE 80
CMD ["bun", "run", "start"]

# Build stage
FROM base AS build
RUN bun run build

# Prod stage – serve the built SPA with nginx
FROM nginx:alpine AS prod
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

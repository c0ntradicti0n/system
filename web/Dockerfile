# Common base stage
FROM node:18 as base

WORKDIR /app

RUN npm install -g bun

COPY package*.json ./

RUN bun install

COPY . .

# Development stage
FROM base as dev

WORKDIR /app

ENV PORT=80

CMD  bun start  | awk '{gsub(/\033c/,"") system("")}1'

# Build stage
FROM base as build

RUN bun run build

# Stage 2: Serve the React application
FROM node:18-alpine as prod

RUN apk add --no-cache curl
RUN curl https://bun.sh/install |  bash

# Install serve
RUN bun install -g serve

# Copy build files from the builder stage
COPY --from=build /app/build /app

WORKDIR /app

CMD serve -p 80 -s . | awk '{gsub(/\033c/,"") system("")}1'

EXPOSE 80

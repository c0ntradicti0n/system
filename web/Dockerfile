# Common base stage
FROM node:18 as base

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

# Development stage
FROM base as dev

WORKDIR /app

ENV PORT=80

CMD ["npm", "start"]

# Build stage
FROM base as build

RUN npm run build

# Stage 2: Serve the React application
FROM node:14-alpine as prod

# Install serve
RUN npm install -g serve

# Copy build files from the builder stage
COPY --from=build /app/build /app

WORKDIR /app

CMD ["serve", "-p", "80", "-s", "."]

EXPOSE 80

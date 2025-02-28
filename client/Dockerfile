# STAGE 1 [Builder]
############################
# Specifies a base image, a specific version of the Node.js image.
#
# This stage is used to build the application.
FROM oven/bun:latest AS builder
# FROM node:22.12.0-alpine3.19 AS builder

# Creates a working directory in the Docker image.
# The application will be placed here.
WORKDIR /usr/src/app

# Copies the package.json and package-lock.json (or npm-shrinkwrap.json) to the app directory.
# This is done separately from copying the rest of the app's code to leverage Docker's cache.
# As a result, the Docker image layer with installed dependencies is only rebuilt when these files change.
COPY bun.lock .
COPY package.json .
# COPY package*.json .

# Runs the npm install command to install the dependencies specified in the package.json file.
#
# Using "npm ci" for production build.
RUN bun install
# RUN npm ci

# Copies the rest of the application code to the app directory.
COPY src/ src/
COPY index.html .
COPY tsconfig.json .
COPY vite.config.ts .

# Runs the build script set in the package.json file.
RUN bun run build
# RUN npm run build


# STAGE 2 [Production]
############################
# Specifies a base image, a specific version of the Nginx image.
FROM nginx:stable-alpine

# Copies the "dist" directory from the builder stage to the app directory.
COPY --from=builder /usr/src/app/dist /usr/share/nginx/html

# Exposes a port that the application listens on.
EXPOSE 80

# Defines the command to run the application.
CMD ["nginx", "-g", "daemon off;"]

FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json .
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build --configuration=production

FROM nginx:1.25-alpine
COPY --from=builder /app/dist/swif_validator_frontend/browser /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80

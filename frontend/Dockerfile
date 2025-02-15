# Use Node.js as base image
FROM node:18-alpine


WORKDIR /app


COPY package*.json ./


RUN npm install


COPY . .


ENV NODE_OPTIONS=--openssl-legacy-provider


RUN npm run build


RUN npm install -g serve


EXPOSE 3000


CMD ["serve", "-s", "build", "-l", "3000"]
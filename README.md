sh```
docker build -t model-api .
docker run -d --name model-api -p 8000:8000 model-api
```

# Pypp

Python preprocessor

## Docker

```shell
# BUILD
docker build -t pypp . 
# RUN
# -v binds a mount from a local directory to one inside the container
# -rm removes the container once it is exited/done
docker run --rm -v "$(pwd)/data:/app/data" pypp -i ./path/to/file.json -o ./path/to/file.json -keys example1 example2
```

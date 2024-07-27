# Dataset labeling and validation
```
docker run -it --name label-studio -d -p 8080:8080 -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/datasets/yolo/ -v "$PWD/dataset/label-studio:/label-studio/data" -v "$PWD/dataset/yolo-format:/datasets/yolo/cv" heartexlabs/label-studio:latest
 ```

<p> Jump in the container </p>
```
docker exec -ti label-studio bash
```

<p> and run this command to import the dataset </p>
```
label-studio-converter import yolo -i /datasets/yolo/cv -o /label-studio/data/output.json --image-root-url "/data/local-files/?d=cv/images/"
```



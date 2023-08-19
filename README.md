Setup

Within the project directory run:
```
python3 -m venv venv
source bin/activate
pip install -r requirements.txt 
```

Download the trained model checkpoint and unzip in the project root
```
https://temp-bucket-map-challenge.s3.eu-west-1.amazonaws.com/content.zip
```

To start the inference endpoint, run:
```
flask --app inference run
```

Format the input data as a json object. input.json is formatted as a list of all rows relating to a given map_id (in the example, it's map_id: 116533).

```
curl -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/predict -d @input.json
```

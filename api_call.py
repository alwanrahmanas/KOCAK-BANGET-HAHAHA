# fungsi inference

def inference_ngrok(path_model, data):
    import requests
    import json

    binary_model = path_model['binary']
    binary_url = binary_model['multiclass']
    headers = {'Content-Type': 'application/json'}
    
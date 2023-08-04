from potassium import Potassium, Request, Response
import time
from transformers import pipeline
import torch

app = Potassium("my_app")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    model = pipeline('fill-mask', model='bert-base-uncased', device=device)

    time.sleep(1)
    print("ran init")
    time.sleep(1)
   
    context = {
        "model": model
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    model = context.get("model")
    outputs = model(prompt)

    print("ran handler")

    return Response(
        json = {"outputs": outputs[0]}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()

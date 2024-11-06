from gluoncv.model_zoo import get_model
import matplotlib.pyplot as plt
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv import utils
from PIL import Image
import io
import flask

app = flask.Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST":
        if flask.request.files.get("img"):
            try:
                # Leer y transformar la imagen
                img = Image.open(io.BytesIO(flask.request.files["img"].read())).convert('RGB')
                transform_fn = transforms.Compose([
                    transforms.Resize(32),
                    transforms.CenterCrop(32),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ])
                img = transform_fn(nd.array(img))

                # Cargar el modelo preentrenado
                net = get_model('cifar_resnet20_v1', classes=10, pretrained=True)

                # Hacer predicción
                pred = net(img.expand_dims(axis=0))
                class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                               'dog', 'frog', 'horse', 'ship', 'truck']
                ind = nd.argmax(pred, axis=1).astype('int')
                prediction = {
                    "class": class_names[ind.asscalar()],
                    "probability": float(nd.softmax(pred)[0][ind].asscalar())
                }
                
                return flask.jsonify(prediction)

            except Exception as e:
                # Manejo de errores
                return flask.jsonify({"error": str(e)}), 400

    return flask.jsonify({"error": "Invalid request"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0')

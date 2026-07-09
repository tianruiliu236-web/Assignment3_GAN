from flask import Flask, jsonify
import torch
from torchvision import transforms
import base64
from io import BytesIO

from models import Generator


app = Flask(__name__)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

generator = Generator(noise_dim=100).to(device)
generator.load_state_dict(torch.load("checkpoints/gan_generator.pth", map_location=device))
generator.eval()


@app.route("/")
def home():
    return jsonify({
        "message": "GAN API is running"
    })


@app.route("/generate-digit", methods=["GET"])
def generate_digit():
    with torch.no_grad():
        noise = torch.randn(1, 100).to(device)
        fake_image = generator(noise)

        fake_image = (fake_image + 1) / 2
        fake_image = fake_image.squeeze().cpu()

        image = transforms.ToPILImage()(fake_image)

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({
        "message": "Generated handwritten digit image",
        "image_base64": encoded_image
    })


if __name__ == "__main__":
    app.run(debug=True)
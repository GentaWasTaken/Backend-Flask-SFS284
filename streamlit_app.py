from flask import Flask, request, jsonify, redirect, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image 
import numpy as np 
from io import BytesIO
from PIL import Image
import os
app = Flask(__name__)
CORS(app)

try:
    model = load_model("flower_classifier_model.h5")
except Exception as e:
    raise ValueError(f"Error loading model: {str(e)}")

LABELS = [
    'Pink primrose', 'Hard-leaved pocket orchid', 'Canterbury bells', 'Sweet pea', 'English marigold', 'Tiger lily',
    'Moon orchid', 'Bird of paradise', 'Monkshood', 'Globe thistle', 'Snapdragon', 'Colt\'s foot', 'King protea',
    'Spear thistle', 'Yellow iris', 'Globe-flower', 'Purple coneflower', 'Peruvian lily', 'Balloon flower',
    'Giant white arum lily'
]
DETAILS = [
    'Bunga primrose merupakan salah satu bunga pertama yang mekar di musim semi. Namanya sebenarnya berasal dari kata Latin \'primus\', yang berarti \'pertama\'. Bunga primrose merah muda memiliki makna kewanitaan, keanggunan, kemudaan, dan pembaruan.', 
    'Hard-leaved pocket orchid atau anggrek kantong berdaun keras adalah nama lain dari anggrek selop perak (Paphiopedilum micranthum). Bunga ini mekar pada akhir musim dingin hingga awal musim panas, dan hanya menghasilkan satu bunga per perbungaan. Berbeda dengan anggrek Paph. malipoense, anggrek Paph. micranthum tidak beraroma.', 
    'Bunga Canterbury bells (Campanula medium) adalah tanaman berbunga yang berasal dari Eropa Selatan dan termasuk dalam genus Campanula. Bunga ini memiliki bentuk lonceng yang mencolok dan tatakan menarik. Dalam floriografi , tanaman ini melambangkan rasa syukur, atau keyakinan dan keteguhan.', 
    'Bunga sweet pea (Lathyrus odoratus) adalah bunga yang memiliki bentuk unik dan menarik, serta aroma harum yang manis. Bunga ini merupakan tanaman asli dari Keluarga Pulsa China dan tersebar di wilayah Mediterania.', 
    'Bunga marigold merupakan salah satu jenis bunga yang memiliki warna kuning cerah dan sering digunakan sebagai tanaman hias, terutama di acara pernikahan. Nama \'Marigold\' berasal dari kata Inggris kuno \'ymbglidegold\'. Artinya “emas di seluruh dunia”. Bunga marigold dikenal dengan keindahan bunganya yang bermekaran.', 
    'Bunga lili harimau (Lilium lancifolium) adalah bunga yang memiliki ciri khas berwarna oranye dengan bintik-bintik hitam yang menyerupai bulu harimau. Bunga ini berasal dari Asia, tepatnya Tiongkok, Jepang, Korea, dan Timur Jauh Rusia.',
    'Anggrek bulan (Phalaenopsis amabilis) merupakan jenis bunga anggrek asli Indonesia. Anggrek bulan masuk dalam keluarga/famili Orchidaceae atau yang lebih dikenal sebagai keluarga anggrek. Bunga anggrek mampu bertahan cukup lama dalam kondisi lingkungan yang bagus. Bahkan, bunga anggrek dapat bertahan hidup lebih dari dua bulan.', 
    'Bunga cenderawasih, bunga bangau atau isigude (Strelitzia reginae)[1] adalah spesies tumbuhan berbunga yang berasal dari Afrika Selatan. Tanaman tahunan yang selalu hijau, dibudidayakan secara luas untuk bunganya yang dramatis. Di daerah beriklim sedang itu adalah tanaman hias yang populer.', 
    'Monkshood adalah tanaman tahunan tegak setinggi 2 hingga 4 kaki dengan bunga runcing berwarna biru-ungu yang indah yang muncul pada pertengahan hingga akhir musim panas. Bunga berbentuk helm yang khas menyerupai tudung jubah biarawan, sehingga mendapat nama umum. Semua bagian tanaman ini beracun, terutama akarnya yang bulat, dan harus ditanam dengan hati-hati, terutama di dekat kebun sayur dan tempat anak-anak bermain.', 
    'Globe thistle adalah tanaman tahunan yang tumbuh cepat dan tampak kontemporer dengan bunga berwarna biru, ungu, atau putih yang menambah warna mencolok pada taman perbatasan musim panas.', 
    'Tanaman Antirrhinum majus dikenal sebagai bunga naga atau snapdragon karena kemiripannya bunganya kepada wajah seekor naga yang membuka dan menutup mulutnya ketika bunganya ditekan. Mereka merupakan bunga asli daerah berbatu di sekita Eropa, Amerika Serikat, dan Afrika Utara. Snapdragon mempunyai batang yang tinggi dengan bunga berwarna yang cerah yang akan mekar ketika cuaca dan suhu sudah dingin.', 
    'Coltsfoot adalah tanaman herba menahun yang menyebar melalui biji dan rimpang . Tussilago sering ditemukan dalam koloni yang terdiri dari puluhan tanaman. Bunganya, yang secara kasat mata menyerupai dandelion, memiliki daun bersisik pada batangnya yang panjang di awal musim semi.', 
    'Protea cynaroides, juga disebut king protea (dari bahasa Afrikaans : koningsprotea, bahasa Xhosa : isiQwane sobukumkani ), adalah tanaman berbunga. Tanaman ini merupakan anggota khas Protea, yang memiliki kepala bunga terbesardalam genusnya. Spesies ini juga dikenal sebagai giant protea, honeypot atau king sugar bush. Tanaman ini tersebar luas di wilayah barat daya dan selatan Afrika Selatan di wilayah fynbos.',
    'Bunga thistle merupakan bunga dari family Asteraceae yang memiliki karakteristik daun berduri di sekitar bunganya ataupun memiliki duri-duri tajam di batang tanamannya yang merupakan adaptasi untuk melindunginya dari hewan-hewan herbivora.', 
    'Bunga iris (Neomarica longifolia; sinonim: Trimezia longifolia) adalah terna (herb) yang berkembang biak dengan menggunakan rhizome (rimpang) dan termasuk ke dalam keluarga Iridaceae. Tumbuhan ini yang mempunyai bunga berwarna kuning cerah dengan corak hitam di bagian tengah (bentuknya seperti bunga anggrek). Tumbuhan ini berasal dari Afrika Barat dan Amerika Selatan, namun juga dapat ditemukan di Indonesia. Salah satu tempat persebarannya adalah Gunung Prau.', 
    'Bunga kenop (Gomphrena globosa) adalah spesies tumbuhan berbunga dari genus Gomphrena; tanaman ini merupakan terna semusim, dan umumnya dimanfaatkan sebagai tanaman hias dan dapat digunakan sebagai teh bunga.', 
    'Bunga kerucut ungu merupakan tanaman tahunan herba dalam keluarga Asteraceae (bunga aster) yang berasal dari Amerika Serikat bagian tengah dan timur. Nama genusnya dalam bahasa Yunani berarti \'berduri\' dan nama spesiesnya berarti ungu kemerahan. Tanaman ini dapat tumbuh setinggi 3 hingga 4 kaki dan menghasilkan bunga berwarna ungu kemerahan yang matang di awal musim panas hingga pertengahan musim gugur.', 
    'Alstroemeria, yang umumnya dikenal sebagai Bunga Lili Peru atau Bunga Lili Suku Inca, berasal dari Amerika Selatan, khususnya Brasil, Argentina, dan Chili. Mirip dengan bunga lili dengan bunga berbentuk terompet yang menarik, bunga ini tidak tumbuh dari umbi sejati melainkan dari akar umbi yang menyebar ke luar, sehingga ukuran tanaman bertambah besar setiap tahunnya.', 
    'Gomphocarpus physocarpus, umumnya dikenal sebagai bunga balon, tanaman balon, semak kapas balon, bola uskup, kepala paku, atau tanaman angsa, adalah spesies dogbane . Tanaman ini berasal dari Afrika tenggara, tetapi telah dinaturalisasi secara luas. Ini sering digunakan sebagai tanaman hias.',
    'Giant white arum lily atau Zantedeschia aethiopica \'White Giant\' adalah pilihan bunga lili calla besar yang tingginya mencapai 7 kaki dan menyebar hingga memenuhi area seluas 4 kaki. Dedaunan hijaunya berbintik-bintik putih. Bunganya berwarna putih dengan tangkai kuning seperti paku. Tanaman ini hanya dapat mencapai tinggi maksimumnya di daerah beriklim hangat, di mana mereka tidak mati sepenuhnya di musim dingin.'
]

@app.route('/cdn/image/<index>/<filename>')
def serve_image(index, filename):
    image_folder = 'valid/valid/{}/'.format(index)
    return send_from_directory(image_folder, filename)

@app.route('/')
def welcome():
    return redirect("https://github.com/GENTA7740", code=302)

@app.route('/flower', methods=['GET'])
def get_flower():
    flowers = []
    for index, flower in enumerate(LABELS):
        
        image_folder = 'valid/valid/{}/'.format(index + 1)
        image_files = os.listdir(image_folder)
        image_files = [f for f in image_files if f.endswith(('.jpg', '.jpeg', '.png'))]
        send_from_directory(image_folder, image_files[0])
        image_link = f'http://0.0.0.0:5000/cdn/image/{index + 1}/{image_files[0]}'
        flowers.append({
            "flower_name": flower,
            "image_link": image_link
        })

    return jsonify({"flowers": flowers}), 200

@app.route('/flower/<int:index>', methods=['GET'])
def get_flower_detail(index):
    image_folder = 'valid/valid/{}/'.format(index + 1)
    image_files = os.listdir(image_folder)
    image_files = [f for f in image_files if f.endswith(('.jpg', '.jpeg', '.png'))]
    send_from_directory(image_folder, image_files[0])
    image_link = f'http://0.0.0.0:5000/cdn/image/{index + 1}/{image_files[0]}'
    

    return jsonify({
        "flower_name": LABELS[index],
        "flower_detail": DETAILS[index],
        "image_link": image_link
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    try:
        img = Image.open(BytesIO(file.read()))
        img = img.resize((128, 128))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        prediction = model.predict(img)
        predicted_indexed = int(np.argmax(prediction))
        predicted_class = LABELS[predicted_indexed]
        confidence = float(np.max(prediction))
        image_folder = 'valid/valid/{}/'.format(predicted_indexed + 1)
        image_files = os.listdir(image_folder)
        image_files = [f for f in image_files if f.endswith(('.jpg', '.jpeg', '.png'))]

        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence,
            "image_files": image_files,
            "image_index": predicted_indexed + 1
        }), 200

    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

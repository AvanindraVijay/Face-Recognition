import bz2
import os
import dlib
def download_model():
    model_url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    model_path = 'shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(model_path):
        import bz2
        import requests
        print("Downloading model...")
        response = requests.get(model_url)
        with open('temp.dat.bz2', 'wb') as f:
            f.write(response.content)
        with bz2.BZ2File('temp.dat.bz2') as fr, open(model_path, 'wb') as fw:
            fw.write(fr.read())
        os.remove('temp.dat.bz2')
    else:
        print("Model already exists.")

download_model()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

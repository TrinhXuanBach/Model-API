import pickle

import uvicorn
from fastapi import FastAPI, UploadFile, File
from pydub import AudioSegment
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow import keras

from utls import audio_url, save_audio, fake_db, extract_feature, test, name_pretrain

app = FastAPI()


def predict():
    with open(name_pretrain, "rb") as file:
        ss = pickle.load(file)
    feature = ss.transform(np.array([test({"url": audio_url})]))
    preds = keras.models.load_model("model").predict_classes(feature)
    return fake_db[preds[0].astype(int)]


@app.post("/files/")
def create_file(
        fileb: UploadFile = File(...)
):
    save_audio(fileb)
    result = predict()
    return {
        "name": result
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

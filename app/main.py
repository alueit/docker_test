from typing import Annotated
import os
import json

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from starlette.status import HTTP_303_SEE_OTHER

from werkzeug.utils import secure_filename
from datetime import datetime

import src.preprocessing as preprocessing
import src.scoring as scoring
from fastapi.templating import Jinja2Templates

app = FastAPI()

root = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(directory="templates")

ALLOWED_EXTENSIONS = set(['csv'])


def generate_page(html: str):
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <title>Churn Prediction</title>
</head>
<body bgcolor="#A0BEC4">
<h1>Анализ оттока абонентов</h1>

<h3>Загрузить файл в формате .csv</h1>
<form action="/uploadfile/" enctype="multipart/form-data" method="post">
<input name="files" type="file">
<input type="submit" value="Отправить">
</form>
<br><br>

{html}

</body>
</html>
    """

@app.get("/feature_importance/{filename}")
def json_feature_importance(filename: str):
    with open(f"output/{filename}".replace("csv", "json"), encoding='utf-8') as fh:
        data = json.load(fh)
    return data

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/download/{filename}")
def download_file(filename: str):
  return FileResponse(f'output/{filename}')

@app.get("/done/{filename}")
def done(filename: str):
    location = os.path.join('output', filename)
    download_link = f"<a href=/download/{filename}>Ссылка на скачивание файла с ответами</a>"
    img_graph = f"<img src=/img/{filename.replace('csv', 'png')}>"
    json_link = f"<a href=/feature_importance/{filename}>JSON влияния признаков</a>"
    html_content = generate_page(download_link + "<br><br>" + json_link + "<br><br>" + img_graph)
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/done/{filename}", response_class=HTMLResponse)
async def done(request: Request, filename: str):
    return templates.TemplateResponse(
        request=request, name="download.html", context={"filename": filename}
    )

@app.get("/")
async def main():
    html_content = generate_page("")
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/uploadfile/")
async def create_upload_files(
    files: Annotated[
        list[UploadFile], File(description="Multiple files as UploadFile")
    ],
):
    uploaded_file = files[0]
    if allowed_file(uploaded_file.filename):
        filename = secure_filename(uploaded_file.filename)
        new_filename = f'{filename.split(".")[0]}_{str(datetime.now())}.csv'.replace(" ", "_")
        save_location = os.path.join('input', new_filename)
        with open(save_location, "wb+") as file_object:
            file_object.write(uploaded_file.file.read())
        
        input_df = preprocessing.import_data(save_location)

        # Run preprocessing
        preprocessed_values = preprocessing.run_preproc(input_df)

        # Run scorer to get submission file for competition
        submission = scoring.make_pred(preprocessed_values, save_location, save_location.replace('input', 'output'))
        submission.to_csv(save_location.replace('input', 'output'), index=False)

        return RedirectResponse(url=f"/done/{new_filename}",status_code=HTTP_303_SEE_OTHER)

@app.get("/img/{filename}")
async def read_image(filename: str):
    return FileResponse(f"output/{filename}")
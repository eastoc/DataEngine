import uvicorn 
from fastapi import FastAPI
from typing import List
from data_model import ImageTagRequest, ImageTagRequestBase64
from onnx_caption_service import inference_tags_batch, inference_tags_url, inference_tags_base64, inference_tags_local_file
import log_util

app = FastAPI()
logger = log_util.get_logger(__name__)


@app.post("/inference/image_captioner")
async def image_captioner(req:  List[dict]):
    logger.info(f"inferenceImageCaptioner - req: {req}")
    return inference_tags_batch(req)

@app.post("/inference/imageTag")
async def image_tag(req:  ImageTagRequest):
    logger.info(f"inferenceImageTag - image_url: {req.image_url}")
    return inference_tags_url(req)

@app.post("/inference/imageTagBase64")
async def image_tag_base64(req: ImageTagRequestBase64):
    return inference_tags_base64(req.image, req.suffix)

@app.post("/inference/imageTagLocalFile")
async def image_tag_local_file(req: ImageTagRequest):
    return inference_tags_local_file(req.image_url)

if __name__ == '__main__':
    uvicorn.run(app="caption_server:app", host='0.0.0.0', port=9528, reload=True)
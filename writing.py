import json
import time
import erniebot
import hashlib

from typing import Optional, List, Dict
import os
import numpy as np
import faiss
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.pdf import PDFMinerLoader
from langchain.document_loaders.word_document import Docx2txtLoader

erniebot.api_type = 'aistudio'
erniebot.access_token = 'YOUR_AISTUDIO_ACCESS_TOKEN'

BASE_PROMPT = """
帮我撰写一篇文章。
要求: 段落过渡清晰自然，不要使用“首先”、“其次”、“再者”、“总之”等词语
输出格式: markdown"""

SUPPORTED_FILE_FORMATS = ["pdf", "docx"]
MAX_CONTEXT_SIZE = 2000

# 模型生成的向量。对于ernie-text-embedding模型，向量维度为384。
# 参考 https://ernie-bot-agent.readthedocs.io/zh-cn/latest/sdk/api_reference/embedding/#_2
VECTOR_DIM = 384


class VectorStore:

    def __init__(self, data: List[Dict]):
        self.data = data
        self.full_text = '\n'.join([v['text'] for v in data])
        self.faiss_index = faiss.IndexFlatL2(VECTOR_DIM)

        for v in data:
            self.faiss_index.add(np.array([v['embedding']], dtype=np.float32))

    def retrieval(self, query: str, top_k: int = 8) -> List[str]:
        if len(self.full_text) < MAX_CONTEXT_SIZE:
            return [self.full_text]

        response = erniebot.Embedding.create(
            model="ernie-text-embedding",
            input=[query]
        )

        query_embedding = response.get_result()[0]

        scores, indices = self.faiss_index.search(
            np.array([query_embedding], dtype=np.float32), top_k
        )

        rets: List[str] = []

        for j, i in enumerate(indices[0]):
            rets.append(self.data[i]['text'])

        return rets


def progress_packet(text, progress):
    return {
        'type': 'progress',
        'value': {
            'text': text,
            'progress': progress,
        }
    }


def create_vectors(file_path: str):
    if file_path.endswith("pdf"):
        doc_loader = PDFMinerLoader(file_path)
    else:
        doc_loader = Docx2txtLoader(file_path)

    yield progress_packet('split document', 30)

    chunks = doc_loader.load_and_split(text_splitter=CharacterTextSplitter(
        chunk_size=300, chunk_overlap=0, separator=''
    ))

    progress = 50
    yield progress_packet('embedding file', progress)

    texts = [
        v.page_content.strip()
        for v in chunks
    ]

    if len(texts) == 0:
        yield progress_packet('finished', progress)
        yield VectorStore([])
        return

    last = len(texts) - 1
    buffer = []
    embeddings = []
    each_text_progress = 40 / len(texts)

    for idx in range(len(texts)):
        buffer.append(texts[idx])
        if len(buffer) != 16 and idx != last:
            continue

        response = erniebot.Embedding.create(
            model="ernie-text-embedding",
            input=buffer
        )

        for embedding in response.get_result():
            embeddings.append(embedding)

        # 更新进度条
        progress += each_text_progress * len(buffer)
        yield progress_packet('embedding file', progress)
        # 清理缓存
        buffer.clear()

    vectors: List[Dict] = [
        {
            'text': texts[i],
            'embedding': embeddings[i]
        }
        for i in range(len(embeddings))
    ]

    yield progress_packet('creating vector store', progress)

    vector_cache = f'{file_path}.vec'
    with open(vector_cache, 'w+', encoding='utf-8') as f:
        f.write(json.dumps(vectors, ensure_ascii=False))

    yield from load_vectors(vector_cache)


def load_vectors(file_path: str):
    with open(file_path, "r", encoding='utf-8') as f:
        json_str = f.read()

    vector_data = json.loads(json_str)
    yield progress_packet('creating vector store', 95)

    yield VectorStore(vector_data)

    yield progress_packet('finished!', 100)


def writing_engine(
        access_token: str,
        title: str,
        writing_content: str = "",
        language_style: str = "",
        reference_file: Optional[UploadedFile] = None
):
    erniebot.access_token = access_token
    # [解析文件]
    context: str = ''
    if reference_file is not None:
        try:
            ext = os.path.splitext(reference_file.name)[1][1:].lower()
        except:
            ext = reference_file.name.split(".")[-1]

        yield progress_packet('read file', 1)
        raw_data = reference_file.read()

        yield progress_packet('calc sha1', 10)

        file_sha1 = hashlib.sha1(raw_data).hexdigest().upper()
        file_path = f"./upload_files/{file_sha1}.{ext}"

        # 判断是否有缓存
        if not os.path.exists(file_sha1) and not os.path.exists(f'{file_sha1}.vec'):
            with open(file_path, mode="wb") as f:
                f.write(raw_data)
            vector_factory_func = create_vectors
        else:
            vector_factory_func = load_vectors

        vector_store = None
        for packet in vector_factory_func(file_path):
            if type(packet) is dict:
                yield packet
            else:
                vector_store = packet

        retrieval_texts = vector_store.retrieval(
            title + writing_content
        )
        context = ''.join(retrieval_texts)
        context = context.replace("\n\n", "\n")

    # [推理]
    title = title.strip()
    language_style = language_style.strip()
    writing_content = writing_content.strip()

    # [prompt]
    writing_prompt = BASE_PROMPT + f"\n标题: {title}"
    if language_style != '':
        writing_prompt += f'\n语言风格: {language_style}'
    if writing_content != '':
        writing_prompt += f'\n写作要点: {writing_content}'
    if context != '':
        writing_prompt += f'\n背景信息: {context}'

    print(writing_prompt)

    stream = erniebot.ChatCompletion.create(
        messages=[
            {'role': 'user', 'content': writing_prompt}
        ],
        model='ernie-3.5',
        stream=True,
    )
    for packet in stream:
        result = packet.get_result()
        for ch in result:
            yield {
                'type': 'predict',
                'value': ch
            }
            time.sleep(0.02)

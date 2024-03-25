import time
import os

import streamlit as st

from writing import writing_engine

if not os.path.exists("./upload_files"):
    os.mkdir("./upload_files")

st.set_page_config(
    page_title='文思写作助手 | 基于自有文件的写作增强',
    menu_items={
        'About': 'Powered by SodaAI'
    },
)
st.sidebar.title('✍ 文思写作助手')
st.sidebar.markdown("""
基于自有文件的写作增强
""")


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False

title = st.sidebar.text_input(
    "主题",
    placeholder="例如：人工智能在教育领域应用",
)

writing_content = st.sidebar.text_area(
    "写作要点",
    height=200,
    placeholder="请输入写作要点，如：关键词、概要信息、目标受众等",
)

language_style = st.sidebar.selectbox(
    label='语言风格',
    options=[
        '正式',
        '严肃',
        '轻松',
        '简洁',
        '幽默',
        '喜悦',
        '激情'
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "上传一个参考文件，让生成内容更准确",
    type=["pdf", "docx"],
    help='',
    on_change=clear_submit,
)


def writing_article():
    if title.strip() == '':
        st.warning('请输入标题!')
        return

    engine = writing_engine(
        "AIStudio-AccessToken",
        title,
        writing_content=writing_content,
        language_style=language_style,
        reference_file=uploaded_file,
    )

    article_content: str = ''

    bar = st.sidebar.progress(0)

    for packet in engine:
        if packet['type'] == 'progress':
            args = packet['value']
            bar.progress(int(args['progress']))
        else:
            article_content += packet['value']
            break

    bar.empty()

    with st.markdown(body=''):
        for packet in engine:
            article_content += packet['value']
            st.write(article_content)


btn = st.sidebar.button("🚀生成文章", on_click=writing_article)

st.sidebar.markdown(
    """
[🥰体验文思助手高级版](http://wensi.sodabot.cn/api/v1/share/?code=lvX3nj&platform=web)
"""
)


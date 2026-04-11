import sys, warnings, os
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, '/home/hanyuu/rag_project')

from loaders.word_loader import WordLoader
from splitters.markdown_splitter import split_documents
from chains.rag_chain import build_vectorstore, build_rag_chain

DOC_PATH = '/home/hanyuu/rag_project/data/raw/【保密】240万吨加氢裂化装置工艺技术规程（试行）-部分章节.doc'

print("Step 1: 加载文档（Word → PDF → MinerU → chunks）...")
docs = WordLoader(DOC_PATH, source_name='hydro_manual').load()
chunks = split_documents(docs)
print(f"  文档块: {len(chunks)}")

print("Step 2: 建向量库...")
vectorstore = build_vectorstore(chunks, collection_name="hydro_manual")

print("Step 3: 问答测试...")
chain = build_rag_chain(collection_name="hydro_manual")

questions = [
    "加氢裂化装置的主要反应条件是什么？",
    "装置开工前需要哪些准备工作？",
    "反应器的操作温度和压力范围是多少？",
]

for q in questions:
    print(f"\n{'='*50}")
    print(f"问：{q}")
    result = chain.invoke(q)
    print(f"答：{result['answer']}")
    print("来源：")
    for doc in result['source_docs'][:2]:
        print(f"  [p.{doc.metadata.get('page')}] {doc.page_content[:80]}...")

import os
from difflib import get_close_matches
import glob
import json
import time
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 1. SETUP & DATA LOADING
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, request_timeout=30, max_retries=2)

def load_dataset(dataset_path):
    print(f"Loading files from {dataset_path}...")
    docs = []
    # Xử lý trường hợp thư mục lồng nhau dataset/dataset
    file_pattern = os.path.join(dataset_path, "**", "*.txt")
    for filepath in glob.glob(file_pattern, recursive=True):
        with open(filepath, "r", encoding="utf-8") as f:
            docs.append(f.read())
    print(f"Loaded {len(docs)} documents.")
    return docs

# 2. ENTITY & RELATION EXTRACTION (INDEXING)
def extract_triples(docs):
    print("Extracting triples using LLM...")
    prompt_template = """
Bạn là chuyên gia trích xuất thông tin từ văn bản về các công ty công nghệ.

NHIỆM VỤ: Trích xuất các bộ ba (Subject, Relation, Object) từ văn bản bên dưới.

QUY TẮC BẮT BUỘC:
1. Subject và Object PHẢI là Named Entity thuộc một trong các loại sau:
   - COMPANY: Tên công ty (ví dụ: OpenAI, Google, Apple, Microsoft, Tesla)
   - PERSON: Tên người thật cụ thể (ví dụ: Sam Altman, Elon Musk, Satya Nadella)
   - PRODUCT: Tên sản phẩm hoặc dịch vụ cụ thể (ví dụ: ChatGPT, iPhone, Azure, Gmail)
   - TECHNOLOGY: Tên công nghệ cụ thể (ví dụ: GPT-4, CUDA, Large Language Model)
2. TUYỆT ĐỐI KHÔNG trích xuất những thực thể chung chung như: "users", "investors", "analysts", "people", "government", "markets", "the company", các con số, phần trăm, hoặc các cụm từ mô tả.
3. Relation phải ngắn gọn, in hoa (tối đa 4 từ). Ví dụ: FOUNDED_BY, CEO_OF, ACQUIRED, PARTNERS_WITH, DEVELOPED, INVESTED_IN, SUBSIDIARY_OF.
4. Nếu văn bản không có thực thể đáp ứng quy tắc trên, hãy trả về danh sách rỗng.

VÍ DỤ:
Input: "Sam Altman, CEO of OpenAI, said Microsoft invested $10B to help develop ChatGPT."
Output: {{"triples": [["Sam Altman", "CEO_OF", "OpenAI"], ["Microsoft", "INVESTED_IN", "OpenAI"], ["OpenAI", "DEVELOPED", "ChatGPT"]]}}

Văn bản cần phân tích: {text}

Trả về JSON với key "triples":
    """
    
    parser = JsonOutputParser()
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    
    all_triples = []
    total_tokens = 0
    # Xử lý tuần tự để tránh rate limit (có thể dùng ThreadPoolExecutor để nhanh hơn)
    for i, text in enumerate(docs):
        if not text.strip(): continue
        try:
            # Rút gọn text nếu quá dài
            result = chain.invoke({"text": text[:2000]})
            triples = result.get("triples", [])
            for t in triples:
                if len(t) == 3:
                    # Chuẩn hóa (strip) và in hoa relation
                    all_triples.append((str(t[0]).strip(), str(t[1]).strip().upper(), str(t[2]).strip()))
            print(f"Doc {i+1}/{len(docs)}: Extracted {len(triples)} triples.")
        except Exception as e:
            print(f"Error processing doc {i+1}: {e}")
            
    # Khử trùng lặp (Deduplication)
    unique_triples = list(set(all_triples))
    print(f"Total unique triples extracted: {len(unique_triples)}")
    return unique_triples

# 3. GRAPH CONSTRUCTION
def build_graph(triples, output_img="knowledge_graph.png"):
    print("Building NetworkX Graph...")
    G = nx.DiGraph()
    for sub, rel, obj in triples:
        G.add_edge(sub, obj, label=rel)
        
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.7)
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
    
    # Draw edge labels (relations)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    
    plt.title("Tech Company Knowledge Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"Graph visualization saved to {output_img}")
    return G

# 4. FLAT RAG SETUP
def build_flat_rag(docs):
    print("Building Flat FAISS RAG...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    langchain_docs = [Document(page_content=t) for t in docs if t.strip()]
    chunks = text_splitter.split_documents(langchain_docs)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

def query_flat_rag(question, retriever):
    relevant_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = f"""Dựa vào thông tin sau đây, hãy trả lời câu hỏi:
    Ngữ cảnh: {context}
    Câu hỏi: {question}
    Trả lời:"""
    
    response = llm.invoke(prompt)
    return response.content

# 5. GRAPHRAG QUERYING
def find_best_matching_node(entity, G):
    """Tìm node tương ứng trong graph bằng fuzzy matching 3 lớp."""
    nodes = list(G.nodes())
    if not nodes:
        return None

    # Lớp 1: Khớp chính xác (không phân biệt hoa thường)
    for node in nodes:
        if entity.lower() == str(node).lower():
            return node

    # Lớp 2: Fuzzy matching (cho phép sai khác nhỏ như "Inc.", "Corp.")
    node_lower_map = {str(n).lower(): n for n in nodes}
    close = get_close_matches(entity.lower(), list(node_lower_map.keys()), n=1, cutoff=0.75)
    if close:
        return node_lower_map[close[0]]

    # Lớp 3: Substring matching (dự phòng)
    for node in nodes:
        if entity.lower() in str(node).lower() or str(node).lower() in entity.lower():
            return node

    return None


def query_graph_rag(question, G):
    # a. Trích xuất thực thể chính từ câu hỏi
    extract_prompt = f"""Trích xuất TÊN THỰC THỂ CHÍNH (tên công ty, người, sản phẩm, hoặc công nghệ) từ câu hỏi sau.
Chỉ trả về đúng tên thực thể bằng tiếng Anh, không thêm gì khác. Ví dụ: "OpenAI", "Sam Altman", "ChatGPT".
Câu hỏi: {question}
Thực thể:"""
    entity_response = llm.invoke(extract_prompt)
    main_entity = entity_response.content.strip().strip('"').strip("'")
    print(f"  [GraphRAG] Extracted entity: '{main_entity}'")

    # Tìm node tốt nhất bằng fuzzy matching 3 lớp
    matched_node = find_best_matching_node(main_entity, G)
    print(f"  [GraphRAG] Matched node: '{matched_node}'")

    if not matched_node:
        context = f"Không tìm thấy thực thể '{main_entity}' trong đồ thị tri thức."
    else:
        # b. Duyệt đồ thị (2-hop traverse)
        neighborhood = []
        # 1-hop outgoing
        for neighbor in G.neighbors(matched_node):
            rel = G.edges[matched_node, neighbor]['label']
            neighborhood.append(f"{matched_node} --[{rel}]--> {neighbor}")
            # 2-hop
            for n2 in G.neighbors(neighbor):
                rel2 = G.edges[neighbor, n2]['label']
                neighborhood.append(f"{neighbor} --[{rel2}]--> {n2}")

        # 1-hop incoming (đảo chiều)
        for predecessor in G.predecessors(matched_node):
            rel = G.edges[predecessor, matched_node]['label']
            neighborhood.append(f"{predecessor} --[{rel}]--> {matched_node}")

        # c. Textualization
        context = "\n".join(list(set(neighborhood)))

    # d. Gửi cho LLM trả lời
    prompt = f"""Bạn là trợ lý AI trả lời câu hỏi dựa trên Đồ thị Tri thức (Knowledge Graph).
Dưới đây là các mối quan hệ liên quan được trích xuất từ đồ thị:

{context}

Dựa VÀO các thông tin trên, hãy trả lời câu hỏi sau một cách đầy đủ và chính xác.
Nếu thông tin trong đồ thị không đủ để trả lời, hãy nói rõ phần nào bạn biết và phần nào không.

Câu hỏi: {question}
Trả lời:"""

    response = llm.invoke(prompt)
    return response.content

# 6. EVALUATION
def evaluate_systems(flat_retriever, G):
    print("Running Evaluation Benchmark...")
    questions = [
        # ZEEKR - node có nhiều kết nối nhất
        "ZEEKR là công ty gì và hoạt động trong lĩnh vực nào?",
        "Doanh thu của ZEEKR trong Q1 2024 là bao nhiêu?",
        "ZEEKR có mối quan hệ gì với Zeekr Intelligent Technology Holding Limited?",
        # Tesla
        "Tesla có vai trò gì trong ngành xe điện?",
        "Elon Musk liên quan đến công ty nào?",
        # NVIDIA
        "Jensen Huang là ai và liên quan đến công ty nào?",
        "NVIDIA NIM là gì?",
        # Nikola Corporation
        "Nikola Corporation tập trung vào lĩnh vực gì?",
        "Nikola Corporation có kế hoạch gì ở Bắc Mỹ?",
        # Polestar
        "Chi phí bán hàng của Polestar là bao nhiêu?",
        # Thị trường EV và chính sách
        "Chính quyền Biden đã cam kết đầu tư bao nhiêu vào năng lượng sạch?",
        "Trung Quốc có vị trí như thế nào trong ngành sản xuất xe điện?",
        "Mối quan hệ giữa Mỹ và Trung Quốc ảnh hưởng thế nào đến ngành EV?",
        # EIA (U.S. Energy Information Administration)
        "EIA cung cấp những loại thông tin gì?",
        "EIA có những công cụ nào để truy cập dữ liệu năng lượng?",
        # VinFast
        "VinFast là công ty đến từ quốc gia nào?",
        # BYD
        "BYD có mối quan hệ gì trong đồ thị tri thức?",
        # Năng lượng và môi trường
        "SEIA là tổ chức gì và hoạt động trong lĩnh vực nào?",
        "Dữ liệu khí nhà kính (Greenhouse gas data) được EIA sử dụng như thế nào?",
        # Multi-hop
        "Mối liên hệ giữa Nikola Corporation và thị trường Bắc Mỹ là gì?"
    ]

    
    results = []
    start_time = time.time()
    
    for q in questions:
        print(f"Q: {q}")
        flat_ans = query_flat_rag(q, flat_retriever)
        graph_ans = query_graph_rag(q, G)
        results.append({
            "Question": q,
            "Flat RAG": flat_ans.replace('\n', ' '),
            "GraphRAG": graph_ans.replace('\n', ' ')
        })
        
    end_time = time.time()
    
    # Save results to markdown
    with open("benchmark_results.md", "w", encoding="utf-8") as f:
        f.write("# Benchmark Results: Flat RAG vs GraphRAG\n\n")
        f.write("| Câu hỏi | Flat RAG | GraphRAG |\n")
        f.write("|---|---|---|\n")
        for r in results:
            f.write(f"| {r['Question']} | {r['Flat RAG']} | {r['GraphRAG']} |\n")
            
        f.write("\n## Cost Analysis\n")
        f.write(f"- Thời gian thực thi truy vấn (20 câu hỏi): {round(end_time - start_time, 2)} giây.\n")
        f.write("- Token usage: Phụ thuộc vào dữ liệu. GraphRAG tối ưu hơn Flat RAG về Context length vì chỉ gửi các node liên quan (2-hop) thay vì gửi toàn bộ chunk lớn.\n")
        
    print("Evaluation saved to benchmark_results.md")

if __name__ == "__main__":
    dataset_path = "dataset"
    docs = load_dataset(dataset_path)

    # Ưu tiên dùng filtered_triples.json nếu đã có, không cần extract lại
    if os.path.exists("filtered_triples.json"):
        print("Found filtered_triples.json, loading directly (skipping LLM extraction)...")
        with open("filtered_triples.json", "r", encoding="utf-8") as f:
            triples = json.load(f)
        print(f"Loaded {len(triples)} filtered triples.")
    elif os.path.exists("extracted_triples.json"):
        print("Found extracted_triples.json, loading directly...")
        with open("extracted_triples.json", "r", encoding="utf-8") as f:
            triples = json.load(f)
        print(f"Loaded {len(triples)} triples.")
    else:
        triples = extract_triples(docs)
        with open("extracted_triples.json", "w", encoding="utf-8") as f:
            json.dump(triples, f, ensure_ascii=False, indent=2)

    G = build_graph(triples)
    flat_retriever = build_flat_rag(docs)

    evaluate_systems(flat_retriever, G)
    print("Pipeline completed successfully!")

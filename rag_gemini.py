import os
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Carrega as variáveis de ambiente
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("A variável de ambiente GOOGLE_API_KEY não está configurada.")
genai.configure(api_key=GOOGLE_API_KEY)

# --- 2. Nossa "Base de Conhecimento" (lendo de arquivo) ---
documents = []

# Dados de aves
try:
    # Abre o arquivo para leitura, garantindo que ele será fechado automaticamente
    with open("dados/aves.txt", "r", encoding="utf-8") as f:
        # Lê cada linha do arquivo como um item separado na lista de documentos
        # .strip() remove espaços em branco extras e quebras de linha
        documents.extend([line.strip() for line in f if line.strip()]) # Ignora linhas vazias
except FileNotFoundError:
    print("Erro: O arquivo 'dados/aves.txt' não foi encontrado. Certifique-se de que ele está no mesmo diretório do script.")
    # Saia ou defina uma lista de documentos de fallback
    documents.append("Nenhum dado encontrado.", "Por favor, crie o arquivo 'dados/aves.txt'.")

# Dados de peixes
try:
    with open("dados/peixes.txt", "r", encoding="utf-8") as f:
        documents.extend([line.strip() for line in f if line.strip()])
except FileNotFoundError:
    documents.append("Nenhum dado encontrado", "Por favor, crie o arquivo 'dados/peixes.txt'. ")

# --- 3. Função para gerar Embeddings ---
def get_embedding(text):
    try:
        response = genai.embed_content(model="models/embedding-001", content=text)
        return response['embedding']
    except Exception as e:
        print(f"Erro ao gerar embedding: {e}")
        return None

document_embeddings = []
for doc in documents:
    embedding = get_embedding(doc)
    if embedding:
        document_embeddings.append(embedding)
    else:
        document_embeddings.append(np.zeros(768))

document_embeddings = np.array(document_embeddings)

# --- 4. Função de Recuperação (Retrieval) ---
def retrieve_relevant_documents(query, top_k=2):
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return []

    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    most_similar_indices = similarities.argsort()[-top_k:][::-1]
    relevant_docs = [documents[i] for i in most_similar_indices]
    return relevant_docs

# --- 5. Função para Geração Aumentada (Augmented Generation) ---
def generate_rag_response(query):
    relevant_docs = retrieve_relevant_documents(query)

    if not relevant_docs:
        print("Nenhum documento relevante encontrado. Tentando responder sem contexto adicional.")
        prompt = f"Responda à seguinte pergunta: {query}"
    else:
        context = "\n".join(relevant_docs)
        prompt = f"""
        Você é um especialista em aves e peixes. Use o seguinte contexto para responder à pergunta.
        Se a resposta não puder ser encontrada no contexto fornecido, diga que não sabe.

        Contexto:
        {context}

        Pergunta: {query}
        Resposta:
        """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Erro ao chamar a API Gemini: {e}")
        return "Desculpe, não consegui gerar uma resposta no momento."

# --- Exemplo de Uso ---
if __name__ == "__main__":
    print("Bem-vindo ao seu assistente RAG (especialista em aves e peixes) !")
    print("Digite 'sair' para encerrar.")

    while True:
        user_query = input("\nSua pergunta: ")
        if user_query.lower() == 'sair':
            break

        response = generate_rag_response(user_query)
        print(f"Assistente: {response}")
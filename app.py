from flask import Flask, render_template, request
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import string

app = Flask(__name__)

# ==========================================
# 1. CARREGAR OS DADOS
# ==========================================

# Carregar os Artigos Médicos
try:
    with open('artigos_medicos.json', 'r', encoding='utf-8') as f:
        artigos = json.load(f)
except FileNotFoundError:
    artigos = []
    print("Aviso: Ficheiro 'artigos_medicos.json' não encontrado.")

# Carregar o Dicionário
try:
    with open('dicionario_unificado.json', 'r', encoding='utf-8') as f:
        dicionario_dados = json.load(f)
except FileNotFoundError:
    dicionario_dados = []
    print("Aviso: Ficheiro 'dicionario_unificado.json' não encontrado.")


# ==========================================
# 2. CONFIGURAR O MOTOR DE BUSCA (TF-IDF)
# ==========================================

# Juntar a doença alvo e o conteúdo para ser mais fácil de encontrar (Resolve o problema da Anemia)
documentos = [(artigo.get("doenca_alvo", "") + " " + artigo.get("conteudo", "")) for artigo in artigos] if artigos else []

# Configurar o modelo
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documentos) if documentos else None


# ==========================================
# 3. ROTAS DA APLICAÇÃO WEB
# ==========================================

@app.route('/')
def index():
    # Página inicial clean
    return render_template('index.html')

@app.route('/dicionario')
def dicionario():
    # Vai buscar a letra que o utilizador escolheu no URL (por defeito começa no 'A')
    letra_escolhida = request.args.get('letra', 'A').upper()
    
    # Filtra o dicionário gigante para mostrar APENAS os termos que começam por essa letra
    termos_filtrados = [
        entrada for entrada in dicionario_dados 
        if entrada.get('termo', '') and entrada['termo'].upper().startswith(letra_escolhida) and entrada.get('definicao')
    ]
    
    # Para o site ser super rápido, limitamos aos primeiros 100 termos dessa letra
    termos_filtrados = termos_filtrados[:100]
    
    # Gera uma lista com as letras de A a Z para enviar para os botões do HTML
    alfabeto = list(string.ascii_uppercase)
    
    return render_template('dicionario.html', termos=termos_filtrados, alfabeto=alfabeto, letra_atual=letra_escolhida)


@app.route('/pesquisa', methods=['GET', 'POST'])
def pesquisa():
    resultados_pesquisa = []
    query_utilizador = ""
    
    if request.method == 'POST':
        query_utilizador = request.form.get('query', '')
        
        if query_utilizador and tfidf_matrix is not None:
            # Transforma a pergunta do utilizador em números usando o modelo
            query_vec = vectorizer.transform([query_utilizador])
            
            # Calcula a similaridade (de 0 a 1)
            similaridades = cosine_similarity(query_vec, tfidf_matrix).flatten()
            
            # Ordena TODOS os resultados do maior para o menor
            todos_indices_ordenados = similaridades.argsort()[::-1]
            
            for idx in todos_indices_ordenados:
                score = similaridades[idx]
                if score > 0.01: # Se a relevância for maior que 1% (ignora lixo)
                    resultados_pesquisa.append((artigos[idx], round(score * 100, 2)))
                    
    # Renderiza a página HTML passando os resultados
    return render_template('pesquisa.html', query=query_utilizador, resultados=resultados_pesquisa)


# Iniciar o Servidor
if __name__ == '__main__':
    app.run(debug=True)
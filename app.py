from flask import Flask, render_template, request, redirect
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import string

app = Flask(__name__)

# ==========================================
# 1. CARREGAR OS DADOS
# ==========================================

# Carregar o super ficheiro de artigos que criaste
try:
    with open('artigos_medicos_unificados.json', 'r', encoding='utf-8') as f:
        artigos = json.load(f)
except FileNotFoundError:
    artigos = []
    print("Aviso: Ficheiro 'artigos_medicos_unificados.json' não encontrado.")

# Carregar o Dicionário Limpo
try:
    with open('dicionario_final.json', 'r', encoding='utf-8') as f:
        dicionario_dados = json.load(f)
except FileNotFoundError:
    dicionario_dados = []
    print("Aviso: Ficheiro 'dicionario_final.json' não encontrado.")

# ==========================================
# 2. CONFIGURAR O MOTOR DE BUSCA (TF-IDF)
# ==========================================
# A LINHA NOVA À PROVA DE BALA FICA AQUI:
documentos_validos = [a for a in artigos if isinstance(a, dict) and a.get("conteudo")]

documentos = [
    (str(artigo.get("doenca_alvo") or "") + " " + str(artigo.get("conteudo") or ""))
    for artigo in documentos_validos
]

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
        if entrada.get('termo', '') and entrada['termo'].upper().startswith(letra_escolhida) and str(entrada.get('definicao', '')).strip()
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

# ==========================================
# 4. OPERAÇÕES CRUD DO DICIONÁRIO
# ==========================================

def salvar_dicionario():
    # Função auxiliar que reescreve o JSON garantindo que não se perde nada
    with open('dicionario_final.json', 'w', encoding='utf-8') as f:
        json.dump(dicionario_dados, f, ensure_ascii=False, indent=2)

@app.route('/adicionar_termo', methods=['POST'])
def adicionar_termo():
    novo_termo = request.form.get('termo', '').strip().lower() # Mantemos a tua regra de tudo minúsculo!
    nova_definicao = request.form.get('definicao', '').strip()
    novo_dominio = request.form.get('dominio', '').strip()
    
    if novo_termo:
        # Verifica se o termo já existe para não criar duplicados
        existe = any(t.get('termo') == novo_termo for t in dicionario_dados)
        
        if not existe:
            nova_entrada = {
                "termo": novo_termo,
                "definicao": nova_definicao,
                "dominios": [novo_dominio] if novo_dominio else [],
                "sinonimos": [],
                "siglas": [],
                "fontes": ["insercao_manual"],
                "traducoes": {}
            }
            dicionario_dados.append(nova_entrada)
            # Volta a ordenar de A-Z
            dicionario_dados.sort(key=lambda x: x.get("termo", ""))
            salvar_dicionario()
            
    # Redireciona para a letra do termo que acabou de adicionar
    letra_redirect = novo_termo[0].upper() if novo_termo else 'A'
    return redirect(f'/dicionario?letra={letra_redirect}')

@app.route('/apagar_termo', methods=['POST'])
def apagar_termo():
    termo_a_apagar = request.form.get('termo', '')
    letra_atual = request.form.get('letra', 'A')
    
    global dicionario_dados
    # Reconstrói a lista ignorando o termo que queremos apagar
    dicionario_dados = [t for t in dicionario_dados if t.get('termo') != termo_a_apagar]
    salvar_dicionario()
    
    # Recarrega a página na mesma letra onde o utilizador estava
    return redirect(f'/dicionario?letra={letra_atual}')

# Iniciar o Servidor
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
from flask import Flask, render_template, request, redirect, jsonify
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import string
import urllib.request
import urllib.parse
import hashlib
import re
import threading
import math

app = Flask(__name__)

# ==========================================
# 1. CARREGAR OS DADOS
# ==========================================

# Carregar o super ficheiro de artigos
try:
    with open('artigos_medicos_unificados.json', 'r', encoding='utf-8') as f:
        artigos = json.load(f)
except FileNotFoundError:
    artigos = []
    print("Aviso: Ficheiro 'artigos_medicos_unificados.json' não encontrado.")

DICIONARIO_FILE = 'dicionario_final.json'

def carregar_dicionario():
    """Carrega sempre o dicionário do disco para garantir dados frescos."""
    try:
        with open(DICIONARIO_FILE, 'r', encoding='utf-8') as f:
            dados = json.load(f)
    except FileNotFoundError:
        dados = []
        print("Aviso: Ficheiro 'dicionario_final.json' não encontrado.")
    
    # Garantir que todas as entradas têm as chaves obrigatórias normalizadas
    for entrada in dados:
        if 'termo' in entrada:
            entrada['termo'] = entrada['termo'].lower().strip()
        entrada.setdefault('dominios', [])
        entrada.setdefault('sinonimos', [])
        entrada.setdefault('siglas', [])
        entrada.setdefault('fontes', [])
        entrada.setdefault('termos_relacionados', [])
        entrada.setdefault('termo_popular', None)
        entrada.setdefault('definicao', "")
        if 'traducoes' not in entrada or not isinstance(entrada['traducoes'], dict):
            entrada['traducoes'] = {lang: None for lang in ['en', 'es', 'fr', 'de', 'pt']}
    return dados

# Carregamento inicial (apenas para o arranque)
dicionario_dados = carregar_dicionario()

# ==========================================
# 2. CONFIGURAR O MOTOR DE BUSCA (TF-IDF & LAZY SBERT)
# ==========================================
documentos_validos = [a for a in artigos if isinstance(a, dict) and a.get("conteudo")]

# Adicionar um ID único e verificar se tem conteúdo válido para QA
for i, artigo in enumerate(documentos_validos):
    artigo['id'] = i
    conteudo = (artigo.get("conteudo") or "").strip()
    conteudo_lc = conteudo.lower()
    # Identificar placeholders comuns
    is_placeholder = (
        len(conteudo) < 80 or 
        "não necessário" in conteudo_lc or 
        "nao necessario" in conteudo_lc or 
        "não aplicável" in conteudo_lc or 
        "nao aplicavel" in conteudo_lc or 
        "no abstract" in conteudo_lc
    )
    artigo['qa_disponivel'] = not is_placeholder

# Incluir o título e categoria de doença no texto a indexar
documentos = [
    (str(artigo.get("titulo") or "") + " " + str(artigo.get("doenca_alvo") or "") + " " + str(artigo.get("conteudo") or ""))
    for artigo in documentos_validos
]

# Stop words combinadas (Inglês do scikit-learn + Português comum)
PT_STOP_WORDS = [
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "não", "nao", "uma", "os", "no", "se", "na", "por", 
    "mais", "ao", "como", "mas", "ele", "das", "à", "seus", "sua", "esta", "pelo", "pela", "até", "ate", "isso", "ela", 
    "entre", "depois", "sem", "mesmo", "aos", "seus", "quem", "nas", "me", "esse", "este", "num", "numa", "suas", 
    "meu", "às", "as", "minha", "têm", "tem", "pelos", "elas", "havia", "qual", "será", "sera", "nós", "nos", "tenho", 
    "lhe", "deles", "essas", "esses", "pelas", "esteio", "fosse", "fomos", "seja"
]
from sklearn.feature_extraction import text
combined_stop_words = list(text.ENGLISH_STOP_WORDS) + PT_STOP_WORDS

class CustomTFIDF:
    def __init__(self, stop_words=None):
        self.stop_words = set(stop_words) if stop_words else set()
        self.vocab = set()
        self.idf_values = {}
        self.doc_vectors = []  # Lista de dicionários: {termo: tf_idf}
        self.doc_norms = []    # Lista de normas L2 dos documentos

    def _tokenizar(self, texto):
        if not texto:
            return []
        import unicodedata
        # Remover acentuação convertendo caracteres unicode para NFD
        texto_norm = "".join(c for c in unicodedata.normalize('NFD', texto.lower()) if unicodedata.category(c) != 'Mn')
        texto_limpo = re.sub(r'[^\w\s]', ' ', texto_norm)
        tokens = [w.strip() for w in re.split(r'\s+', texto_limpo) if w.strip()]
        return [t for t in tokens if t not in self.stop_words]

    def fit(self, corpus):
        N = len(corpus)
        if N == 0:
            return
        
        docs_tokens = [self._tokenizar(doc) for doc in corpus]
        self.vocab = set(term for doc in docs_tokens for term in doc)
        
        # Calcular IDF usando log base 10: idf(t,D) = log10(N/counter)
        for term in self.vocab:
            df = sum(1 for doc in docs_tokens if term in doc)
            self.idf_values[term] = math.log10(N / df) if df > 0 else 0.0

        # Calcular vetores TF-IDF e normas dos documentos
        for doc in docs_tokens:
            if not doc:
                self.doc_vectors.append({})
                self.doc_norms.append(0.0)
                continue
                
            tf_values = {}
            for term in doc:
                tf_values[term] = tf_values.get(term, 0) + 1
            # tf(t,d) = count(t) / total words (d)
            tf_values = {k: v / len(doc) for k, v in tf_values.items()}
            
            doc_vector = {}
            for term, tf_val in tf_values.items():
                doc_vector[term] = tf_val * self.idf_values.get(term, 0.0)
                
            self.doc_vectors.append(doc_vector)
            norm = math.sqrt(sum(val ** 2 for val in doc_vector.values()))
            self.doc_norms.append(norm)

    def search(self, query):
        query_tokens = self._tokenizar(query)
        if not query_tokens or not self.doc_vectors:
            return [0.0] * len(self.doc_vectors)
            
        query_tf = {}
        for term in query_tokens:
            query_tf[term] = query_tf.get(term, 0) + 1
        query_tf = {k: v / len(query_tokens) for k, v in query_tf.items()}
        
        query_vector = {}
        for term, tf_val in query_tf.items():
            if term in self.idf_values:
                query_vector[term] = tf_val * self.idf_values[term]
                
        query_norm = math.sqrt(sum(val ** 2 for val in query_vector.values()))
        if query_norm == 0.0:
            return [0.0] * len(self.doc_vectors)
            
        similaridades = []
        for idx, doc_vector in enumerate(self.doc_vectors):
            doc_norm = self.doc_norms[idx]
            if doc_norm == 0.0:
                similaridades.append(0.0)
                continue
                
            dot_product = sum(query_vector[term] * doc_vector[term] for term in query_vector if term in doc_vector)
            sim = dot_product / (query_norm * doc_norm)
            similaridades.append(sim)
            
        return similaridades

# Configurar o modelo TF-IDF customizado de acordo com aula11.py
custom_tfidf = CustomTFIDF(stop_words=combined_stop_words)
custom_tfidf.fit(documentos)

# Lazy Loading de SBERT
_sbert_model = None
EMBEDDINGS_CACHE_FILE = 'artigos_embeddings.npy'

def get_sbert_model():
    global _sbert_model
    if _sbert_model is None:
        print("A carregar modelo SBERT (medlink-bi-encoder)...")
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("lfcc/medlink-bi-encoder")
    return _sbert_model

def calcular_hash_documentos():
    hasher = hashlib.sha256()
    for doc in documentos:
        hasher.update(doc.encode('utf-8'))
    return hasher.hexdigest()

def get_sbert_embeddings():
    hash_atual = calcular_hash_documentos()
    hash_file = 'artigos_embeddings_hash.txt'
    
    rebuild = True
    if os.path.exists(EMBEDDINGS_CACHE_FILE) and os.path.exists(hash_file):
        try:
            with open(hash_file, 'r', encoding='utf-8') as f:
                saved_hash = f.read().strip()
            if saved_hash == hash_atual:
                rebuild = False
        except Exception:
            pass
            
    if not rebuild:
        return np.load(EMBEDDINGS_CACHE_FILE)
    else:
        model = get_sbert_model()
        print("A gerar/atualizar embeddings SBERT para os artigos (isto ocorre apenas uma vez)...")
        embeddings = model.encode(documentos, show_progress_bar=True)
        np.save(EMBEDDINGS_CACHE_FILE, embeddings)
        try:
            with open(hash_file, 'w', encoding='utf-8') as f:
                f.write(hash_atual)
        except Exception as e:
            print(f"Erro ao salvar hash dos embeddings: {e}")
        print("Embeddings SBERT guardados em cache.")
        return embeddings


# Pré-carregar o SBERT em segundo plano ao iniciar o servidor para evitar latência na primeira pesquisa
def precarregar_sbert_background():
    try:
        print("A pré-carregar embeddings e modelo SBERT em background...")
        get_sbert_embeddings()
        get_sbert_model()
        print("SBERT pré-carregado com sucesso em segundo plano!")
    except Exception as e:
        print(f"Erro ao pré-carregar SBERT em background: {e}")

threading.Thread(target=precarregar_sbert_background, daemon=True).start()


# Lazy Loading do Modelo de Question Answering (QA)
_qa_pipelines = {
    'pt': {
        'tokenizer': None, 
        'model': None, 
        'model_name': "./modelo_qa_pt_finetuned" if os.path.exists("./modelo_qa_pt_finetuned") else "pierreguillou/bert-base-cased-squad-v1.1-portuguese", 
        'fallback': "mrm8488/bert-multi-cased-finedtuned-xquad-tydiqa-goldp"
    },
    'en': {
        'tokenizer': None, 
        'model': None, 
        'model_name': "./modelo_qa_en_finetuned" if os.path.exists("./modelo_qa_en_finetuned") else "distilbert-base-cased-distilled-squad", 
        'fallback': "mrm8488/bert-multi-cased-finedtuned-xquad-tydiqa-goldp"
    }
}


def get_qa_pipeline(lang='pt'):
    global _qa_pipelines
    config = _qa_pipelines.get(lang, _qa_pipelines['pt'])
    if config['tokenizer'] is None or config['model'] is None:
        print(f"A carregar modelo de Question Answering para '{lang}' ({config['model_name']})...")
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        try:
            config['tokenizer'] = AutoTokenizer.from_pretrained(config['model_name'])
            config['model'] = AutoModelForQuestionAnswering.from_pretrained(config['model_name'])
            print(f"Modelo primário de QA ({config['model_name']}) carregado com sucesso.")
        except Exception as e:
            print(f"Erro ao carregar o modelo primário de QA: {e}. A tentar fallback...")
            fb = config['fallback']
            config['tokenizer'] = AutoTokenizer.from_pretrained(fb)
            config['model'] = AutoModelForQuestionAnswering.from_pretrained(fb)
            print(f"Modelo de fallback de QA ({fb}) carregado com sucesso.")
            
    def executar_qa(question=None, context=None, **kwargs):
        import torch
        tokenizer = config['tokenizer']
        model = config['model']
        # Tokenizar os inputs
        inputs = tokenizer(question, context, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        # Descodificar os tokens de resposta
        answer_tokens = inputs.input_ids[0, answer_start:answer_end]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # Calcular confiança aproximada
        start_probs = torch.softmax(outputs.start_logits, dim=-1)
        end_probs = torch.softmax(outputs.end_logits, dim=-1)
        score = float(start_probs[0, answer_start] * end_probs[0, answer_end - 1])
        
        # Calcular offsets para destacar no texto original
        encoding = tokenizer(question, context, return_offsets_mapping=True)
        offsets = encoding.offset_mapping
        
        char_start = 0
        char_end = 0
        
        if answer_start < len(offsets) and answer_end - 1 < len(offsets):
            sequence_ids = encoding.sequence_ids()
            context_tokens_indices = [i for i, seq_id in enumerate(sequence_ids) if seq_id == 1]
            
            if context_tokens_indices:
                answer_context_tokens = [idx for idx in range(int(answer_start), int(answer_end)) if idx in context_tokens_indices]
                if answer_context_tokens:
                    first_tok = answer_context_tokens[0]
                    last_tok = answer_context_tokens[-1]
                    char_start = offsets[first_tok][0]
                    char_end = offsets[last_tok][1]
                    
        return {
            'answer': answer,
            'score': score,
            'start': char_start,
            'end': char_end
        }
        
    return executar_qa


# ==========================================
# 3. AUXILIARES E ENRIQUECIMENTO AUTOMÁTICO
# ==========================================

def detectar_idioma(texto):
    if not texto:
        return 'pt'
    texto_lc = texto.lower()
    palavras_en = {'the', 'of', 'and', 'is', 'in', 'to', 'with', 'for', 'that', 'by'}
    palavras_pt = {'o', 'a', 'os', 'as', 'de', 'do', 'da', 'em', 'um', 'uma', 'para', 'com'}
    
    tokens = re.findall(r'\b\w+\b', texto_lc)
    count_en = sum(1 for t in tokens if t in palavras_en)
    count_pt = sum(1 for t in tokens if t in palavras_pt)
    
    return 'en' if count_en > count_pt else 'pt'

def detectar_idioma_pergunta(pergunta):
    if not pergunta:
        return 'pt'
    pergunta_lc = pergunta.lower()
    
    palavras_pt = {
        'qual', 'quais', 'como', 'onde', 'quando', 'quem', 'porque', 'porquê', 'o que', 
        'sintomas', 'tratamento', 'doente', 'idade', 'paciente', 'diagnóstico', 'causa',
        'é', 'são', 'doença', 'infecção', 'infeção', 'medicamento', 'terapia', 'de', 'do', 'da'
    }
    palavras_en = {
        'what', 'who', 'where', 'when', 'why', 'how', 'which', 'is', 'are', 'the', 'of',
        'symptoms', 'treatment', 'patient', 'age', 'diagnosis', 'cause', 'disease', 
        'infection', 'medication', 'therapy', 'with', 'about', 'does', 'did', 'was'
    }
    
    tokens = re.findall(r'\b\w+\b', pergunta_lc)
    count_pt = sum(1 for t in tokens if t in palavras_pt)
    count_en = sum(1 for t in tokens if t in palavras_en)
    
    if count_pt == count_en:
        if re.search(r'[áéíóúâêôãõçã]', pergunta_lc):
            return 'pt'
        for t in tokens:
            if t in ['what', 'how', 'why', 'who', 'where']:
                return 'en'
            if t in ['qual', 'quais', 'como', 'onde', 'quem']:
                return 'pt'
        return 'pt'
        
    return 'en' if count_en > count_pt else 'pt'

def traduzir_texto(texto, de_lang='auto', para_lang='en'):
    if not texto or not texto.strip():
        return texto
    try:
        texto_enc = urllib.parse.quote(texto.strip())
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={de_lang}&tl={para_lang}&dt=t&q={texto_enc}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            traduzido = "".join([part[0] for part in res_data[0] if part and part[0]])
            return traduzido.strip()
    except Exception as e:
        print(f"Erro ao traduzir de '{de_lang}' para '{para_lang}': {e}")
        return texto

def strings_iguais(s1, s2):
    if not s1 or not s2:
        return s1 == s2
    def normalizar(s):
        return re.sub(r'[^\w\s]', '', s.lower().strip())
    return normalizar(s1) == normalizar(s2)


def buscar_wikipedia(termo):
    """Consulta a API REST do Wikipédia em português para obter o resumo de um termo."""
    try:
        url = f"https://pt.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(termo)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) MedPlatform/1.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            if 'extract' in res_data:
                return res_data['extract']
    except Exception as e:
        print(f"Erro ao pesquisar Wikipédia para '{termo}': {e}")
    return None

def salvar_dicionario(dados):
    """Grava a lista de dados no disco imediatamente e atualiza a variável global."""
    global dicionario_dados
    dados.sort(key=lambda x: x.get("termo", ""))
    with open(DICIONARIO_FILE, 'w', encoding='utf-8') as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    # Sincronizar a variável em memória
    dicionario_dados = dados

# ==========================================
# 4. ROTAS DA APLICAÇÃO WEB
# ==========================================

@app.route('/')
def index():
    # Calcular estatísticas simples para o portal
    num_termos = len(dicionario_dados)
    num_artigos = len(documentos_validos)
    
    dominios_unicos = set()
    for termo in dicionario_dados:
        for dom in termo.get('dominios', []):
            if dom:
                dominios_unicos.add(dom.strip())
    num_dominios = len(dominios_unicos)

    return render_template('index.html', num_termos=num_termos, num_artigos=num_artigos, num_dominios=num_dominios)

@app.route('/dicionario')
def dicionario():
    query_filtro = request.args.get('q', '').strip()
    dominio_filtro = request.args.get('dominio', '').strip()
    letra_escolhida = request.args.get('letra', 'A').upper()
    
    # Carregar dados frescos do disco
    dados_atuais = carregar_dicionario()
    
    termos_filtrados = dados_atuais
    
    # Aplicar filtro de texto
    if query_filtro:
        termos_filtrados = [
            t for t in termos_filtrados
            if query_filtro.lower() in t.get('termo', '')
        ]
        
    # Aplicar filtro de dominio
    if dominio_filtro:
        termos_filtrados = [
            t for t in termos_filtrados
            if dominio_filtro in t.get('dominios', [])
        ]
        
    # Filtro alfabetico: apenas se nao houver filtros de texto ou dominio
    # 'TODOS' mostra tudo (limitado a 200 para performance)
    if not query_filtro and not dominio_filtro:
        if letra_escolhida != 'TODOS':
            termos_filtrados = [
                t for t in termos_filtrados
                if t.get('termo', '').upper().startswith(letra_escolhida)
            ]
    
    # Limitar para performance na renderizacao
    limite = 200 if letra_escolhida == 'TODOS' or query_filtro or dominio_filtro else 100
    termos_exibidos = termos_filtrados[:limite]
    
    alfabeto = list(string.ascii_uppercase)
    
    # Calcular todos os dominios unicos para o painel de filtro
    todos_dominios = set()
    for t in dados_atuais:
        for dom in t.get('dominios', []):
            if dom and dom.strip():
                todos_dominios.add(dom.strip())
    lista_dominios = sorted(todos_dominios)
    
    return render_template(
        'dicionario.html', 
        termos=termos_exibidos, 
        alfabeto=alfabeto, 
        letra_atual=letra_escolhida,
        query_filtro=query_filtro,
        dominio_atual=dominio_filtro,
        lista_dominios=lista_dominios,
        total_filtrados=len(termos_filtrados)
    )

@app.route('/dominios')
def dominios():
    # Manter a rota mas redirecionar para o dicionario
    return redirect('/dicionario')

@app.route('/conceito/<termo>')
def conceito_detalhe(termo):
    termo_norm = termo.lower().strip()
    # Procurar o termo no dicionário
    conceito = next((t for t in dicionario_dados if t.get('termo') == termo_norm), None)
    
    if not conceito:
        # Se não existe, cria uma entrada temporária não persistida para visualização e edição imediata
        conceito = {
            "termo": termo_norm,
            "definicao": "",
            "dominios": [],
            "sinonimos": [],
            "siglas": [],
            "fontes": ["criacao_dinamica"],
            "traducoes": {lang: None for lang in ['en', 'es', 'fr', 'de', 'pt']},
            "termos_relacionados": [],
            "termo_popular": None
        }
    
    # Encontrar sugestões no mesmo domínio
    termos_mesmo_dominio = []
    if conceito.get('dominios'):
        doms = set(conceito['dominios'])
        termos_mesmo_dominio = [
            t for t in dicionario_dados
            if t.get('termo') != termo_norm and any(d in doms for d in t.get('dominios', []))
        ]
        
    return render_template('conceito.html', conceito=conceito, termos_mesmo_dominio=termos_mesmo_dominio)

# ==========================================
# 5. OPERAÇÕES CRUD & ENRIQUECIMENTO
# ==========================================

@app.route('/adicionar_termo', methods=['POST'])
def adicionar_termo():
    novo_termo = request.form.get('termo', '').strip().lower()
    nova_definicao = request.form.get('definicao', '').strip()
    novo_dominio = request.form.get('dominio', '').strip()
    novo_termo_popular = request.form.get('termo_popular', '').strip() or None
    novos_sinonimos = [s.strip().lower() for s in request.form.get('sinonimos', '').split(',') if s.strip()]
    novas_siglas = [s.strip() for s in request.form.get('siglas', '').split(',') if s.strip()]
    novos_relacionados = [t.strip().lower() for t in request.form.get('termos_relacionados', '').split(',') if t.strip()]
    
    if novo_termo:
        # Reler o JSON para ter dados atualizados
        dados_atuais = carregar_dicionario()
        
        # Verificar duplicados
        entrada_existente = next((t for t in dados_atuais if t.get('termo') == novo_termo), None)
        
        if not entrada_existente:
            # Processar domínios (pode vir separado por vírgula)
            dominios_lista = [d.strip() for d in novo_dominio.split(',') if d.strip()] if novo_dominio else ['Termo Clínico']
            
            nova_entrada = {
                "termo": novo_termo,
                "definicao": nova_definicao,
                "dominios": dominios_lista,
                "sinonimos": novos_sinonimos,
                "siglas": novas_siglas,
                "fontes": ["insercao_manual"],
                "traducoes": {
                    "en": request.form.get('trans_en', '').strip() or None,
                    "es": request.form.get('trans_es', '').strip() or None,
                    "fr": request.form.get('trans_fr', '').strip() or None,
                    "de": request.form.get('trans_de', '').strip() or None,
                    "pt": novo_termo
                },
                "termos_relacionados": novos_relacionados,
                "termo_popular": novo_termo_popular
            }
            dados_atuais.append(nova_entrada)
            salvar_dicionario(dados_atuais)
            
    return redirect(f'/conceito/{novo_termo}')


@app.route('/editar_termo/<termo_original>', methods=['POST'])
def editar_termo(termo_original):
    termo_original_norm = termo_original.lower().strip()
    
    # Reler o JSON para ter dados atualizados e evitar race conditions
    dados_atuais = carregar_dicionario()
    
    # Localizar o termo
    entrada = next((t for t in dados_atuais if t.get('termo') == termo_original_norm), None)
    
    if not entrada:
        # Se não existia, cria um novo
        entrada = {"termo": termo_original_norm, "fontes": []}
        dados_atuais.append(entrada)
        
    # Atualizar campos básicos
    entrada["definicao"] = request.form.get('definicao', '').strip()
    entrada["termo_popular"] = request.form.get('termo_popular', '').strip() or None
    
    # Processar listas separadas por vírgula
    entrada["dominios"] = [d.strip() for d in request.form.get('dominios', '').split(',') if d.strip()]
    entrada["sinonimos"] = [s.strip().lower() for s in request.form.get('sinonimos', '').split(',') if s.strip()]
    entrada["siglas"] = [s.strip() for s in request.form.get('siglas', '').split(',') if s.strip()]
    entrada["termos_relacionados"] = [t.strip().lower() for t in request.form.get('termos_relacionados', '').split(',') if t.strip()]
    
    # Processar traduções
    entrada["traducoes"] = {
        "en": request.form.get('trans_en', '').strip() or None,
        "es": request.form.get('trans_es', '').strip() or None,
        "fr": request.form.get('trans_fr', '').strip() or None,
        "de": request.form.get('trans_de', '').strip() or None,
        "pt": termo_original_norm
    }
    
    if "insercao_manual" not in entrada.get("fontes", []):
        if not entrada.get("fontes"):
            entrada["fontes"] = []
        entrada["fontes"].append("insercao_manual")
        entrada["fontes"] = list(set(entrada["fontes"]))
        
    salvar_dicionario(dados_atuais)
    return redirect(f'/conceito/{termo_original_norm}')

@app.route('/apagar_termo', methods=['POST'])
def apagar_termo():
    termo_a_apagar = request.form.get('termo', '').strip().lower()
    letra_atual = request.form.get('letra', 'A')
    
    # Reler JSON, filtrar o termo apagado, e guardar
    dados_atuais = carregar_dicionario()
    dados_atuais = [t for t in dados_atuais if t.get('termo') != termo_a_apagar]
    salvar_dicionario(dados_atuais)
    
    return redirect(f'/dicionario?letra={letra_atual}')

@app.route('/enriquecer_termo/<termo>', methods=['POST'])
def enriquecer_termo(termo):
    termo_norm = termo.lower().strip()
    
    # Reler JSON e procurar o termo
    dados_atuais = carregar_dicionario()
    entrada = next((t for t in dados_atuais if t.get('termo') == termo_norm), None)
    
    resumo_wiki = buscar_wikipedia(termo_norm)
    
    if resumo_wiki:
        if not entrada:
            # Criar entrada se não existir
            entrada = {
                "termo": termo_norm,
                "definicao": "",
                "dominios": ["Termo Clínico"],
                "sinonimos": [],
                "siglas": [],
                "fontes": [],
                "traducoes": {lang: None for lang in ['en', 'es', 'fr', 'de', 'pt']},
                "termos_relacionados": [],
                "termo_popular": None
            }
            dados_atuais.append(entrada)
            
        entrada["definicao"] = resumo_wiki
        if "wikipedia" not in entrada.get("fontes", []):
            if not entrada.get("fontes"):
                entrada["fontes"] = []
            entrada["fontes"].append("wikipedia")
            entrada["fontes"] = list(set(entrada["fontes"]))
            
        salvar_dicionario(dados_atuais)
        
    return redirect(f'/conceito/{termo_norm}')

# ==========================================
# 6. ROTAS DE PESQUISA & QUESTION ANSWERING
# ==========================================

def expandir_query(query_original):
    if not query_original:
        return query_original, []
    
    # Tokenizar a query original por palavras
    palavras = [p.strip().lower() for p in re.split(r'[\s,.;:!?]+', query_original) if p.strip()]
    termos_adicionados = set()
    
    # Reler/carregar o dicionario
    dicionario = carregar_dicionario()
    
    # 1. Procurar correspondência exata do termo completo na base de dados
    query_completa = query_original.strip().lower()
    for entrada in dicionario:
        termo_dict = entrada.get('termo', '')
        if termo_dict == query_completa:
            # Adicionar sinónimos
            for sin in entrada.get('sinonimos', []):
                if sin:
                    termos_adicionados.add(sin.lower())
            # Adicionar tradução inglesa
            trad_en = entrada.get('traducoes', {}).get('en')
            if trad_en:
                termos_adicionados.add(trad_en.lower())
            # Adicionar termo popular
            tp = entrada.get('termo_popular')
            if tp:
                termos_adicionados.add(tp.lower())
            # Adicionar siglas
            for sigla in entrada.get('siglas', []):
                if sigla:
                    termos_adicionados.add(sigla.lower())
    
    # 2. Procurar palavras individuais (apenas com comprimento > 3)
    for palavra in palavras:
        if len(palavra) <= 3:
            continue
        for entrada in dicionario:
            termo_dict = entrada.get('termo', '')
            if termo_dict == palavra:
                for sin in entrada.get('sinonimos', []):
                    if sin:
                        termos_adicionados.add(sin.lower())
                trad_en = entrada.get('traducoes', {}).get('en')
                if trad_en:
                    termos_adicionados.add(trad_en.lower())
                tp = entrada.get('termo_popular')
                if tp:
                    termos_adicionados.add(tp.lower())
                for sigla in entrada.get('siglas', []):
                    if sigla:
                        termos_adicionados.add(sigla.lower())
                        
    # Limpar palavras que já existem na query original
    palavras_originais_norm = set(palavras + [query_completa])
    lista_termos_adicionais = sorted([
        t for t in termos_adicionados 
        if t not in palavras_originais_norm
    ])
    
    query_expandida = query_original
    if lista_termos_adicionais:
        query_expandida = query_original + " " + " ".join(lista_termos_adicionais)
        
    return query_expandida, lista_termos_adicionais

@app.route('/pesquisa', methods=['GET', 'POST'])
def pesquisa():
    resultados_pesquisa = []
    # Suportar GET e POST de forma transparente
    query_utilizador = request.values.get('query', '').strip()
    modelo = request.values.get('modelo', 'tfidf')
    
    query_expandida = query_utilizador
    termos_adicionados = []
    
    if query_utilizador:
        # Expansão inteligente da query usando o dicionário
        query_expandida, termos_adicionados = expandir_query(query_utilizador)
        
        if modelo == 'sbert':
            # Pesquisa semântica usando SBERT
            try:
                # Carregar embeddings e modelo
                doc_embeddings = get_sbert_embeddings()
                model = get_sbert_model()
                
                # Codificar a query expandida (melhora correspondência bilingue e sinónimos)
                query_embedding = model.encode([query_expandida])
                
                # Calcular similaridades usando cosine_similarity do scikit-learn
                similaridades = cosine_similarity(query_embedding, doc_embeddings).flatten()
                
                # Ordenar resultados
                todos_indices = similaridades.argsort()[::-1]
                for idx in todos_indices:
                    score = similaridades[idx]
                    if score > 0.05:  # Filtro de relevância mínima (5%)
                        resultados_pesquisa.append((documentos_validos[idx], round(score * 100, 2)))
            except Exception as e:
                print(f"Erro na pesquisa SBERT: {e}")
                # Fallback automático para TF-IDF se SBERT falhar
                modelo = 'tfidf'
        
        # Se for TF-IDF ou se SBERT falhou
        if modelo == 'tfidf':
            similaridades = custom_tfidf.search(query_expandida)
            todos_indices = np.argsort(similaridades)[::-1]
            
            for idx in todos_indices:
                score = similaridades[idx]
                if score > 0.01:  # Filtro de relevância mínima (1%)
                    resultados_pesquisa.append((documentos_validos[idx], round(score * 100, 2)))
                    
    return render_template(
        'pesquisa.html', 
        query=query_utilizador, 
        query_expandida=query_expandida,
        termos_adicionados=termos_adicionados,
        resultados=resultados_pesquisa, 
        modelo=modelo
    )

@app.route('/qa/<int:artigo_id>', methods=['GET', 'POST'])
def qa_detalhe(artigo_id):
    if artigo_id < 0 or artigo_id >= len(documentos_validos):
        return redirect('/pesquisa')
    artigo = documentos_validos[artigo_id]
    
    if not artigo.get('qa_disponivel', True):
        return render_template('qa.html', artigo=artigo, erro="Este artigo não possui um abstract detalhado para análise de Question Answering.")
    
    if request.method == 'POST':
        pergunta = request.form.get('pergunta', '').strip()
        if not pergunta:
            return render_template('qa.html', artigo=artigo, erro="Pergunta inválida.")
        
        contexto = artigo.get('conteudo', '')
        if not contexto or len(contexto.strip()) < 10:
            return render_template('qa.html', artigo=artigo, erro="O artigo não contém conteúdo suficiente para análise.")
            
        try:
            artigo_lang = detectar_idioma(contexto)
            pergunta_lang = detectar_idioma_pergunta(pergunta)
            
            # Validar se a pergunta está na mesma língua do artigo
            if artigo_lang != pergunta_lang:
                if artigo_lang == 'en':
                    erro_msg = "O artigo selecionado está em Inglês. Por favor, faça a sua pergunta em Inglês."
                else:
                    erro_msg = "O artigo selecionado está em Português. Por favor, faça a sua pergunta em Português."
                return render_template('qa.html', artigo=artigo, erro=erro_msg, pergunta=pergunta)
                
            qa = get_qa_pipeline(lang=artigo_lang)
            res = qa(question=pergunta, context=contexto)
            answer = res.get('answer', '')
            score = res.get('score', 0.0)
            start = res.get('start', 0)
            end = res.get('end', 0)
            
            destacado = None
            if answer and start < end and start >= 0 and end <= len(contexto):
                antes = contexto[:start]
                destacado_segmento = contexto[start:end]
                depois = contexto[end:]
                destacado = f"{antes}<mark class='bg-warning text-dark px-1 rounded'>{destacado_segmento}</mark>{depois}"
            
            return render_template(
                'qa.html', 
                artigo=artigo, 
                pergunta=pergunta, 
                resposta=answer, 
                score=score, 
                destacado=destacado
            )
        except Exception as e:
            print(f"Erro no QA: {e}")
            return render_template('qa.html', artigo=artigo, erro=f"Erro interno no processamento: {str(e)}")
            
    return render_template('qa.html', artigo=artigo)


@app.route('/perguntar', methods=['POST'])
def perguntar():
    """Endpoint API para responder a uma pergunta sobre um artigo clínico."""
    data = request.get_json()
    if not data or 'artigo_id' not in data or 'pergunta' not in data:
        return jsonify({'erro': 'Parâmetros em falta. Exigido artigo_id e pergunta.'}), 400
        
    artigo_id = int(data['artigo_id'])
    pergunta = data['pergunta'].strip()
    
    if artigo_id < 0 or artigo_id >= len(documentos_validos):
        return jsonify({'erro': 'Artigo não encontrado.'}), 404
        
    artigo = documentos_validos[artigo_id]
    if not artigo.get('qa_disponivel', True):
        return jsonify({'erro': 'Este artigo não possui um abstract detalhado para análise.'}), 400
        
    contexto = artigo.get('conteudo', '')
    
    if not contexto or len(contexto.strip()) < 10:
        return jsonify({'erro': 'O artigo selecionado não possui conteúdo para análise.'}), 400
        
    try:
        artigo_lang = detectar_idioma(contexto)
        pergunta_lang = detectar_idioma_pergunta(pergunta)
        
        # Validar se a pergunta está na mesma língua do artigo
        if artigo_lang != pergunta_lang:
            return jsonify({'erro': f'A pergunta deve ser feita no mesmo idioma do artigo (idioma exigido: {artigo_lang}).'}), 400
            
        # Obter pipeline de QA (lazy loaded)
        qa = get_qa_pipeline(lang=artigo_lang)
        
        # Executar extração da resposta
        res = qa(question=pergunta, context=contexto)
        answer = res.get('answer', '')
        
        # Retornar resposta estruturada em JSON
        return jsonify({
            'answer': answer,
            'score': float(res['score']),
            'start': int(res['start']),
            'end': int(res['end'])
        })
    except Exception as e:
        print(f"Erro no processamento do Question Answering: {e}")
        return jsonify({'erro': f"Erro interno ao processar QA: {str(e)}"}), 500

# Iniciar o Servidor
if __name__ == '__main__':
    # Nota: Desligamos o use_reloader para evitar carregar o SBERT/QA duas vezes
    # se o utilizador correr em modo debug normal.
    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
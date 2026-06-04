import json
import re
import os
import sys
import urllib.parse
import warnings
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Silenciar aviso de parser HTML para documentos XML
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def limpar_autores(texto):
    if not texto:
        return ""
    
    # 1. Remove ligações ORCID entre parênteses, ex: (https://orcid.org/0000...)
    texto_limpo = re.sub(r'\(https?://orcid\.org/[^\)]+\)', '', texto)
    
    # 2. Remove números de afiliação colados aos nomes, ex: Gonçalves1,2 ou Gomes3
    # Procura dígitos (e vírgulas entre eles) que venham imediatamente a seguir a uma letra
    texto_limpo = re.sub(r'(?<=[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ])\d+(?:,\d+)*', '', texto_limpo)
    
    # 3. Remove qualquer tag HTML (como na versão original)
    texto_limpo = re.sub(r'<[^>]+>', ' ', texto_limpo)
    
    # 4. Corrige espaços duplos e remove espaços desnecessários antes das vírgulas (ex: "Gonçalves , " -> "Gonçalves, ")
    texto_limpo = re.sub(r'\s*,\s*', ', ', texto_limpo)
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()
    
    return texto_limpo

def pesquisar_pubmed(termo, max_resultados=5):
    """
    Pesquisa no PubMed e devolve uma lista de IDs de artigos (PMIDs).
    Usa requests e BeautifulSoup de forma semelhante ao aula5.py.
    """
    termo_encoded = urllib.parse.quote(termo)
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={termo_encoded}&retmax={max_resultados}&retmode=json"
    
    print(f"A pesquisar PubMed por '{termo}'...")
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        data = response.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        return id_list
    except Exception as e:
        print(f"Erro ao pesquisar no PubMed: {e}")
        return []

def extrair_detalhes_artigos(id_list):
    """
    Descarrega e extrai os detalhes dos artigos (título, autores, abstract) via efetch.
    Usa BeautifulSoup para fazer o parsing do XML recebido.
    """
    if not id_list:
        return []
    
    ids = ",".join(id_list)
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={ids}&retmode=xml"
    
    print(f"A descarregar metadados dos artigos (IDs: {ids})...")
    artigos_extraidos = []
    
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        xml_data = response.text
        
        soup = BeautifulSoup(xml_data, "html.parser")
        
        # Como usamos html.parser, as tags XML são lidas em minúsculas
        for article_tag in soup.find_all("pubmedarticle"):
            # 1. Título
            titulo_tag = article_tag.find("articletitle")
            titulo = titulo_tag.get_text().strip() if titulo_tag is not None else "Sem Título"
            
            # 2. Autores
            autores_lista = []
            for author in article_tag.find_all("author"):
                last_name_tag = author.find("lastname")
                fore_name_tag = author.find("forename")
                
                last = last_name_tag.get_text().strip() if last_name_tag is not None else ""
                fore = fore_name_tag.get_text().strip() if fore_name_tag is not None else ""
                
                if last or fore:
                    # Formato: Last Fore
                    autores_lista.append(f"{last} {fore}".strip())
            
            autores = ", ".join(autores_lista) if autores_lista else "Autores não especificados"
            
            # 3. Abstract / Conteúdo
            abstract_texts = []
            for abstract_text in article_tag.find_all("abstracttext"):
                text = abstract_text.get_text().strip()
                if text:
                    abstract_texts.append(text)
            
            conteudo = " ".join(abstract_texts) if abstract_texts else ""
            
            # Ignorar artigos sem conteúdo / abstract
            if not conteudo:
                continue
                
            artigos_extraidos.append({
                "titulo": titulo,
                "autores": autores,
                "conteudo": conteudo
            })
            
        return artigos_extraidos
    except Exception as e:
        print(f"Erro ao extrair detalhes dos artigos: {e}")
        return []

def executar_scraping(doencas_pesquisa=None):
    """
    Executa o processo de scraping completo para várias doenças e guarda no JSON local.
    No final, chama automaticamente a unificação de datasets.
    """
    if doencas_pesquisa is None:
        doencas_pesquisa = ["tuberculosis", "diabetes", "asthma", "hypertension", "malaria"]
        
    ficheiro_artigos = "artigos_medicos.json"
    
    # Carregar artigos existentes para não duplicar títulos
    artigos_existentes = []
    titulos_existentes = set()
    
    if os.path.exists(ficheiro_artigos):
        try:
            with open(ficheiro_artigos, 'r', encoding='utf-8') as f:
                artigos_existentes = json.load(f)
                for art in artigos_existentes:
                    titulos_existentes.add(art.get("titulo", "").lower().strip())
        except Exception as e:
            print(f"Erro ao ler {ficheiro_artigos}: {e}")
            
    novos_artigos_adicionados = 0
    
    for doenca in doencas_pesquisa:
        print(f"\n--- A processar doença: {doenca} ---")
        ids = pesquisar_pubmed(doenca, max_resultados=3)
        artigos = extrair_detalhes_artigos(ids)
        
        for art in artigos:
            titulo_clean = art["titulo"].lower().strip()
            if titulo_clean not in titulos_existentes:
                art["doenca_alvo"] = doenca
                artigos_existentes.append(art)
                titulos_existentes.add(titulo_clean)
                novos_artigos_adicionados += 1
                print(f"Adicionado: {art['titulo'][:60]}...")
            else:
                print(f"Ignorado (já existe): {art['titulo'][:60]}...")
                
    if novos_artigos_adicionados > 0:
        try:
            with open(ficheiro_artigos, 'w', encoding='utf-8') as f:
                json.dump(artigos_existentes, f, ensure_ascii=False, indent=2)
            print(f"\n[OK] Sucesso! Adicionados {novos_artigos_adicionados} novos artigos a '{ficheiro_artigos}'.")
        except Exception as e:
            print(f"Erro ao salvar ficheiro JSON: {e}")
    else:
        print("\nNenhum artigo novo encontrado.")

    # Unifica os datasets no final (seja com novos artigos ou não)
    print("\nA iniciar unificação dos datasets...")
    unificar_datasets()

def unificar_datasets():
    artigos_finais = []

    # ==========================================
    # 1. PROCESSAR O DATASET DO PROFESSOR (PT)
    # ==========================================
    print("A limpar e processar o dataset do professor (dataset_articles.json)...")
    try:
        with open('dataset_articles.json', 'r', encoding='utf-8') as f:
            artigos_base = json.load(f)
            for art in artigos_base:
                artigos_finais.append({
                    "doenca_alvo": art.get("category", "Geral"),
                    "titulo": art.get("title", ""),
                    "autores": limpar_autores(art.get("authors", "")),
                    "conteudo": art.get("abstract", ""),
                    "link": art.get("link", "")
                })
        print(f"-> Sucesso: {len(artigos_base)} artigos do professor processados.")
    except FileNotFoundError:
        print("-> Erro: Ficheiro 'dataset_articles.json' não encontrado.")

    # ==========================================
    # 2. PROCESSAR OS ARTIGOS DO PUBMED (EN)
    # ==========================================
    print("\nA juntar os teus artigos extraídos do PubMed (artigos_medicos.json)...")
    try:
        with open('artigos_medicos.json', 'r', encoding='utf-8') as f:
            artigos_pubmed = json.load(f)
            for art in artigos_pubmed:
                link_gerado = art.get("link", f"https://pubmed.ncbi.nlm.nih.gov/?term={art.get('titulo', '')}")
                
                artigos_finais.append({
                    "doenca_alvo": art.get("doenca_alvo", "Geral"),
                    "titulo": art.get("titulo", ""),
                    "autores": art.get("autores", ""),
                    "conteudo": art.get("conteudo", ""),
                    "link": link_gerado
                })
        print(f"-> Sucesso: {len(artigos_pubmed)} artigos do PubMed adicionados.")
    except FileNotFoundError:
        print("-> Erro: Ficheiro 'artigos_medicos.json' não encontrado.")

    # ==========================================
    # 3. GUARDAR O FICHEIRO UNIFICADO MESTRE
    # ==========================================
    with open('artigos_medicos_unificados.json', 'w', encoding='utf-8') as f:
        json.dump(artigos_finais, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] CONCLUÍDO! O ficheiro 'artigos_medicos_unificados.json' foi criado com {len(artigos_finais)} artigos no total.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--scrape":
        executar_scraping()
    else:
        unificar_datasets()
        print("\n* Dica: Se quiseres descarregar novos artigos do PubMed antes de unificar, corre:")
        print("  python extrairdocumentos.py --scrape")
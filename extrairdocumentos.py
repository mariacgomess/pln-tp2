import json
import re

def limpar_autores(texto):
    if not texto:
        return ""
    # 1. Remove qualquer tag HTML (tudo o que estiver entre < e >)
    texto_limpo = re.sub(r'<[^>]+>', ' ', texto)
    # 2. Esmaga espaços duplos, \n e \t num único espaço
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()
    return texto_limpo

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
                    "autores": limpar_autores(art.get("authors", "")), # Limpeza das tags HTML acontece aqui!
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
                # Se o link não existir no JSON, criamos um na hora para o PubMed
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

    print(f"\n✅ CONCLUÍDO! O ficheiro 'artigos_medicos_unificados.json' foi criado com {len(artigos_finais)} artigos no total.")

if __name__ == "__main__":
    unificar_datasets()
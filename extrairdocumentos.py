import requests
from bs4 import BeautifulSoup
import json
import time

def raspar_varias_doencas():
    # A tua lista de 10 doenças para dar variedade ao Motor de Busca
    lista_doencas = [
        "diabetes", "asthma", "hypertension", "tuberculosis", "alzheimer", "cystic fibrosis",
        "parkinson", "leukemia", "psoriasis", "arthritis", "glaucoma", "anemia"
    ]
    
    artigos_extraidos = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    print("A iniciar o Web Scraping (Bónus) para múltiplas doenças...\n")
    
    for doenca in lista_doencas:
        print(f"A raspar artigos sobre: {doenca}...")
        
        # Vamos pedir os primeiros 10 resultados para garantir que apanhamos pelo menos 6 bons
        url = f"https://pubmed.ncbi.nlm.nih.gov/?term={doenca}&size=10"
        
        try:
            resposta = requests.get(url, headers=headers)
            soup = BeautifulSoup(resposta.text, 'html.parser')
            artigos_html = soup.find_all('article', class_='full-docsum')
            
            contador = 0
            for artigo in artigos_html:
                if contador >= 5: 
                    break
                    
                titulo_tag = artigo.find('a', class_='docsum-title')
                titulo = titulo_tag.text.strip() if titulo_tag else ""
                
                autores_tag = artigo.find('span', class_='docsum-authors')
                autores = autores_tag.text.strip() if autores_tag else ""
                
                snippet_tag = artigo.find('div', class_='full-view-snippet')
                snippet = snippet_tag.text.strip() if snippet_tag else ""
                snippet = " ".join(snippet.split())
                
                # Só guarda se tiver realmente um título e um resumo
                if titulo and snippet:
                    conteudo_completo = f"{titulo}. {snippet}"
                    
                    artigos_extraidos.append({
                        "doenca_alvo": doenca, # Guardamos a tag da doença (dá jeito)
                        "titulo": titulo,
                        "autores": autores,
                        "conteudo": conteudo_completo
                    })
                    contador += 1
            
            # Pausa de 1 segundo entre pesquisas para o site não nos bloquear (boas práticas)
            time.sleep(1) 
            
        except Exception as e:
            print(f"Erro ao pesquisar {doenca}: {e}")

    # Guardar tudo no ficheiro final
    with open('artigos_medicos.json', 'w', encoding='utf-8') as f:
        json.dump(artigos_extraidos, f, ensure_ascii=False, indent=2)
        
    print(f"\nSucesso! Extraídos {len(artigos_extraidos)} artigos no total.")
    print("O ficheiro 'artigos_medicos.json' está pronto!")

if __name__ == "__main__":
    raspar_varias_doencas()
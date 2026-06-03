import json
import wikipedia
import re
import warnings
from langdetect import detect, LangDetectException

# Desligar o aviso do BeautifulSoup (GuessedAtParserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')

wikipedia.set_lang("pt")

def normalizar_ortografia(t):
    t = re.sub(r'ô', 'ó', t)
    t = re.sub(r'ê', 'é', t)
    t = re.sub(r'cç', 'ç', t)
    t = re.sub(r'ct', 't', t)
    return t

def limpar_termo(termo):
    t = str(termo).lower()
    t = re.sub(r'\s+', ' ', t).strip()
    
    t = re.sub(r'\[.*?\]', '', t)
    t = re.sub(r'\(.*?\)', '', t)
    
    t = re.sub(r'\b(pt|br)\b\W*$', '', t, flags=re.IGNORECASE)
    t = t.replace("'", "").replace("?", "")
    t = re.sub(r'\s+', ' ', t).strip()
    
    return normalizar_ortografia(t)

def limpar_lista_strings(lista):
    return list(set([limpar_termo(item) for item in lista if item]))

def e_portugues_estrito(texto):
    if not texto or str(texto).strip() == "":
        return False
        
    try:
        if detect(texto) != 'pt':
            return False
    except LangDetectException:
        return False
        
    texto_lower = " " + texto.lower() + " "
    palavras_proibidas = [
        " the ", " of ", " is ", " and ", " disease ", " with ", " by ", " for ", 
        " el ", " y ", " enfermedad ", " los ", " las ", " del ", " con ", 
        " malaltia ", " els ", " l'", " dels ", " amb " 
    ]
    
    for palavra in palavras_proibidas:
        if palavra in texto_lower:
            return False 
            
    return True

def processar_dataset():
    with open('dicionario_unificado.json', 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    termos_unicos = {}
    termos_atualizados = 0
    definicoes_apagadas = 0
    
    print("🧹 Fase 1: Limpeza Suprema e Unificação a decorrer...")
    
    for entrada in dataset:
        termo_original = entrada.get("termo", "")
        termo_limpo = limpar_termo(termo_original)
        
        if not termo_limpo:
            continue
            
        definicao_atual = entrada.get("definicao", "")
        if definicao_atual and not e_portugues_estrito(definicao_atual):
            entrada["definicao"] = "" 
            definicoes_apagadas += 1

        if termo_limpo in termos_unicos:
            termos_unicos[termo_limpo]["sinonimos"].extend(limpar_lista_strings(entrada.get("sinonimos") or []))
            termos_unicos[termo_limpo]["dominios"].extend(entrada.get("dominios") or [])
            termos_unicos[termo_limpo]["sinonimos"] = list(set(termos_unicos[termo_limpo]["sinonimos"]))
            termos_unicos[termo_limpo]["dominios"] = list(set(termos_unicos[termo_limpo]["dominios"]))
            
            if not termos_unicos[termo_limpo].get("definicao") and entrada.get("definicao"):
                termos_unicos[termo_limpo]["definicao"] = entrada["definicao"]
            continue
            
        entrada["termo"] = termo_limpo
        entrada["sinonimos"] = limpar_lista_strings(entrada.get("sinonimos") or [])
        entrada["dominios"] = list(set(entrada.get("dominios") or []))
        
        termos_unicos[termo_limpo] = entrada

    # Fase 2: Pesquisar na Wikipédia (AGORA COM BARRAS DE PROGRESSO NO TERMINAL!)
    termos_vazios = [k for k, v in termos_unicos.items() if not v.get("definicao") or str(v.get("definicao")).strip() == ""]
    total_vazios = len(termos_vazios)
    
    print(f"\n🌐 Fase 2: Identificados {total_vazios} termos sem definição (vazios ou estrangeiros apagados).")
    print("A iniciar pesquisa na Wikipédia. Isto pode demorar alguns minutos...")
    
    contador = 0
    for termo in termos_vazios:
        contador += 1
        entrada = termos_unicos[termo]
        
        # Imprime o progresso (ex: [15/340] A pesquisar: abcesso...)
        print(f"[{contador}/{total_vazios}] A processar: {termo}...", end="\r")
        
        try:
            resumo = wikipedia.summary(termo, sentences=2)
            if e_portugues_estrito(resumo):
                entrada["definicao"] = resumo
                if isinstance(entrada.get("fontes"), list) and "wikipedia" not in entrada["fontes"]:
                    entrada["fontes"].append("wikipedia")
                termos_atualizados += 1
                # Se encontrar, diz!
                print(f"[{contador}/{total_vazios}] ✅ Encontrado: {termo}                   ")
        except Exception:
            pass

    # Fase 3: Remover quem não tem definição e adicionar Categoria
    print("\n\n📦 Fase 3: A criar o dicionário final e a eliminar termos s4em solução...")
    dataset_final = []
    for entrada in termos_unicos.values():
        definicao = entrada.get("definicao", "")
        
        if definicao and str(definicao).strip() != "":
            if not entrada.get("dominios") or len(entrada["dominios"]) == 0:
                entrada["dominios"] = ["Termo Clínico"]
            dataset_final.append(entrada)

    dataset_final.sort(key=lambda x: x.get("termo", ""))

    with open('dicionario_final.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_final, f, ensure_ascii=False, indent=2)

    print(f"\n--- RELATÓRIO FINAL ---")
    print(f"❌ {definicoes_apagadas} definições ESTRANGEIRAS foram apagadas.")
    print(f"✅ {termos_atualizados} definições em PT adicionadas da Wikipédia.")
    print(f"🧹 TERMOS VAZIOS ELIMINADOS! O dicionário final tem agora {len(dataset_final)} termos.")

if __name__ == "__main__":
    processar_dataset()
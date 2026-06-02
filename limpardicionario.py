import json
import wikipedia
import re

wikipedia.set_lang("pt")

def normalizar_ortografia(t):
    t = re.sub(r'ô', 'ó', t)
    t = re.sub(r'ê', 'é', t)
    t = re.sub(r'cç', 'ç', t)   # acção → ação
    t = re.sub(r'ct', 't', t)   # facto → fato
    return t

def limpar_termo(termo):
    t = str(termo)

    # 1. Minúsculas
    t = t.lower()

    # 2. Espaços invisíveis
    t = re.sub(r'\s+', ' ', t).strip()

    # 3. Marcadores regionais — formato colchete: [Br.] [Pt.] [Pt] [BR]
    t = re.sub(r'\s*\[\s*(pt\.?|br\.?)\s*\]', '', t, flags=re.IGNORECASE)
    # 3b. Formato parêntese: (pt) (br)
    t = re.sub(r'\s*\(\s*(pt|br)\s*\)', '', t, flags=re.IGNORECASE)

    # 4. Marcadores de categoria: (sg), (abrev.), (f), (m), (pop.), etc.
    t = re.sub(r'\s*\(\s*(sg|abrev\.?|arc\.?|pop\.?|cult\.?|col\.?|[mf])\s*\)', '', t, flags=re.IGNORECASE)

    # 5. Colchetes populares/arcaicos: [pop.], [arc.], [cult.]
    t = re.sub(r'\s*\[\s*(pop\.?|arc\.?|cult\.?|col\.?)\s*\]', '', t, flags=re.IGNORECASE)

    # 6. Prefixos numéricos: "35 imunidade natural" → "imunidade natural"
    t = re.sub(r'^\d+\s+', '', t)

    # 7. Parênteses opcionais de letra: "(d)escamação" → "descamação"
    t = re.sub(r'\(([a-záéíóúàãõêôâç])\)', r'\1', t)

    # 8. Aspas simples/duplas
    t = t.replace("'", "").replace('"', '')

    # 9. Interrogações
    t = t.replace("?", "")

    # 10. Parênteses sobreviventes: "(herpes) zóster" → "herpes zóster"
    t = t.replace("(", "").replace(")", "")

    # 11. Normalização ortográfica PT/BR
    t = normalizar_ortografia(t)

    # 12. Limpeza final
    t = re.sub(r'^[\s\-,;\.]+|[\s\-,;\.]+$', '', t)
    t = re.sub(r'\s{2,}', ' ', t).strip()

    return t

def limpar_lista_strings(lista):
    vistas = set()
    resultado = []
    for item in lista:
        limpo = re.sub(r'\s+', ' ', str(item)).strip() if item else ''
        if limpo and limpo not in vistas:
            vistas.add(limpo)
            resultado.append(limpo)
    return resultado

def processar_dataset():
    with open('dicionario_unificado.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    termos_unicos  = {}
    termos_atualizados = 0

    print("A iniciar a Limpeza Suprema (tudo minúsculas, sem pt/br, sem lixo)...")

    for entrada in dataset:
        termo_original = entrada.get("termo", "")
        termo_limpo    = limpar_termo(termo_original)

        if not termo_limpo:
            continue

        # UNIFICAÇÃO: funde domínios, sinónimos, siglas e fontes
        if termo_limpo in termos_unicos:
            existente = termos_unicos[termo_limpo]

            existente["dominios"] = list(set(
                (existente.get("dominios") or []) + (entrada.get("dominios") or [])
            ))
            existente["sinonimos"] = limpar_lista_strings(
                (existente.get("sinonimos") or []) + (entrada.get("sinonimos") or [])
            )
            existente["siglas"] = limpar_lista_strings(
                (existente.get("siglas") or []) + (entrada.get("siglas") or [])
            )
            existente["fontes"] = list(set(
                (existente.get("fontes") or []) + (entrada.get("fontes") or [])
            ))
            if not existente.get("definicao") and entrada.get("definicao"):
                existente["definicao"] = entrada["definicao"]
            if not existente.get("termo_popular") and entrada.get("termo_popular"):
                existente["termo_popular"] = entrada["termo_popular"]
            continue

        # NOVA ENTRADA: limpa campos e regista
        entrada["termo"]     = termo_limpo
        entrada["sinonimos"] = limpar_lista_strings(entrada.get("sinonimos") or [])
        entrada["siglas"]    = limpar_lista_strings(entrada.get("siglas") or [])
        entrada["dominios"]  = list(set(entrada.get("dominios") or []))
        entrada["fontes"]    = list(set(entrada.get("fontes") or []))

        # ENRIQUECIMENTO via Wikipedia
        if not entrada.get("definicao"):
            try:
                resumo = wikipedia.summary(termo_limpo, sentences=2)
                entrada["definicao"] = resumo
                if isinstance(entrada.get("fontes"), list) and "wikipedia" not in entrada["fontes"]:
                    entrada["fontes"].append("wikipedia")
                termos_atualizados += 1
                print(f"Definição adicionada para: {termo_limpo}")
            except Exception:
                pass

        termos_unicos[termo_limpo] = entrada

    # Ordena A-Z e grava no mesmo sítio que o original
    dataset_final = sorted(termos_unicos.values(), key=lambda x: x.get("termo", ""))

    with open('dicionario_unificado.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_final, f, ensure_ascii=False, indent=2)

    print(f"\nConcluído! {termos_atualizados} definições adicionadas.")
    print(f"O ficheiro original tinha {len(dataset)} termos.")
    print(f"O ficheiro final tem {len(dataset_final)} termos únicos, puros e brilhantes.")

if __name__ == "__main__":
    processar_dataset()
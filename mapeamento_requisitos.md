# Mapeamento de Requisitos - Trabalho Prático 2 (PLN)

Este documento descreve a correspondência entre os requisitos do enunciado do **Trabalho Prático 2** e a implementação realizada no código-fonte do projeto. Pode ser utilizado como base estrutural para a redação do relatório técnico em LaTeX e para a preparação dos slides da apresentação oral.

---

## 1. Enriquecimento do Dataset (Tarefa 1)

### Requisitos do Enunciado
* Enriquecer o dataset adicionando conceitos e novos atributos (sinónimos, definições, categorias/domínios, etc.) utilizando fontes externas (websites, dicionários médicos, etc.).

### Implementação no Código
* **Integração com a Wikipédia:** No ficheiro [app.py](file:///c:/Users/Maria/Desktop/Universidade/mestrado/PLN/plneb-2526/Projetos/TP2/pln-tp2/app.py), a função `buscar_wikipedia(termo)` faz pedidos HTTP à API REST da Wikipédia em português (`pt.wikipedia.org/api/rest_v1/page/summary/`) para recolher a definição de forma dinâmica.
* **Inserção Dinâmica:** A rota Flask `@app.route('/enriquecer_termo/<termo>')` invoca a busca na Wikipédia, insere o resumo extraído na base de dados de dicionário e adiciona `"wikipedia"` à lista de `fontes` da palavra.
* **Modelo de Dados do Conceito:** Cada termo no ficheiro `dicionario_final.json` possui um esquema completo com suporte para:
  * `definicao` (Texto)
  * `dominios` (Lista de categorias médicas)
  * `sinonimos` e `siglas` (Listas)
  * `termos_relacionados` (Lista)
  * `traducoes` (Dicionário com traduções mapeadas para `en`, `es`, `fr`, `de`, `pt`)
  * `fontes` (Lista de proveniência dos dados, ex: `wikipedia`, `insercao_manual`)

---

## 2. Plataforma Web (Tarefa 2)

### Requisitos do Enunciado
* Criação de uma ferramenta de visualização de dados para exploração do dataset.
* Navegação adequada recorrendo às relações existentes entre conceitos, ordenação por atributos/categorias.
* Capacidade de atualizar e acrescentar novas informações (CRUD) com persistência definitiva.

### Implementação no Código
* **Servidor Flask ([app.py](file:///c:/Users/Maria/Desktop/Universidade/mestrado/PLN/plneb-2526/Projetos/TP2/pln-tp2/app.py)):** Trata o roteamento da plataforma e a lógica de persistência.
* **Interface Gráfica (Templates HTML):**
  * `index.html`: Estatísticas gerais (número total de termos, categorias e artigos médicos).
  * `dicionario.html`: Listagem de todos os conceitos, com paginação alfabética (A-Z), barra de pesquisa e filtragem avançada por domínio/categoria clínica.
  * `conceito.html`: Página de detalhe de cada conceito, exibindo todos os seus metadados, traduções e uma lista de termos sugeridos no mesmo domínio médico.
* **Operações de Manipulação de Dados (CRUD):**
  * Rota `/adicionar_termo`: Recebe dados do formulário e insere um conceito totalmente novo.
  * Rota `/editar_termo/<termo_original>`: Permite editar os metadados existentes de um termo.
  * Rota `/apagar_termo`: Remove um conceito do dicionário.
* **Persistência em Disco:** A função `salvar_dicionario(dados)` grava as alterações de forma síncrona diretamente no ficheiro JSON `dicionario_final.json`, garantindo que nenhuma alteração se perde caso o servidor reinicie.

---

## 3. Information Retrieval (Tarefa 3)

### Requisitos do Enunciado
* Desenvolvimento de um sistema de IR para realizar pesquisas sobre a coleção de artigos médicos, baseado em TF-IDF ou SBERT.
* **Bónus:** Web scraping para aumentar a coleção de documentos disponíveis.

### Implementação no Código
* **Motor TF-IDF Customizado (Implementado do Zero):**
  * Desenvolvido na classe `CustomTFIDF` em [app.py](file:///c:/Users/Maria/Desktop/Universidade/mestrado/PLN/plneb-2526/Projetos/TP2/pln-tp2/app.py), baseando-se estritamente nas fórmulas do ficheiro `aula11.py`:
    * **Normalização e Tokenização (`_tokenizar`):** Converte para minúsculas, remove pontuação e remove a acentuação (normalização unicode NFD). Filtra também stop-words através da lista bilingue `combined_stop_words`.
    * **Term Frequency (TF):** Mapeia a frequência de cada termo dividindo pela extensão do documento: $TF(t,d) = \frac{\text{count}(t)}{\text{total\_words}(d)}$.
    * **Inverse Document Frequency (IDF):** Calcula a relevância usando o logaritmo de base 10: $IDF(t, D) = \log_{10}\left(\frac{N}{df}\right)$.
    * **Pesquisa por Cosseno:** Representa a query expandida em TF-IDF e calcula a similaridade por cosseno com cada artigo em tempo linear: $\text{Sim}(q, d) = \frac{V_q \cdot V_d}{\|V_q\| \|V_d\|}$.
* **Pesquisa Semântica (SBERT):**
  * Integrada na rota `/pesquisa`, usando o modelo de representação biomédica `"lfcc/medlink-bi-encoder"`.
  * **Otimização de Cache:** Os embeddings dos 901 artigos são calculados e guardados em cache (`artigos_embeddings.npy`) juntamente com o hash do dataset. Nas pesquisas seguintes, o sistema lê o cache em $0.06\text{s}$, evitando lentidão.
  * **Pré-carregamento Assíncrono:** Utiliza uma thread em segundo plano (`precarregar_sbert_background()`) no arranque do Flask para que o modelo SBERT seja carregado na memória RAM em background, eliminando o lag na primeira pesquisa do utilizador.
* **Bónus - Web Scraping:**
  * Implementado no script [extrairdocumentos.py](file:///c:/Users/Maria/Desktop/Universidade/mestrado/PLN/plneb-2526/Projetos/TP2/pln-tp2/extrairdocumentos.py).
  * Faz chamadas HTTP à API do **PubMed** (eutils) para as doenças: *tuberculosis*, *diabetes*, *asthma*, *hypertension* e *malaria*.
  * Utiliza `BeautifulSoup` para fazer o parsing dos abstracts e metadados XML, unificando-os com o dataset principal `dataset_articles.json` no ficheiro final `artigos_medicos_unificados.json` (perfazendo 901 artigos).

---

## 4. Question Answering (Tarefa 4)

### Requisitos do Enunciado
* Integração de um modelo de Question Answering baseado em arquitetura BERT disponível no HuggingFace para responder a perguntas a partir de um artigo selecionado.
* **Bónus:** Fazer o fine-tuning do modelo de QA em vez de usar um modelo pré-treinado existente.

### Implementação no Código
* **Inferência de QA com BERT:**
  * Integrado na rota `/qa/<artigo_id>` e no endpoint API `/perguntar` em [app.py](file:///c:/Users/Maria/Desktop/Universidade/mestrado/PLN/plneb-2526/Projetos/TP2/pln-tp2/app.py).
  * O sistema deteta o idioma do artigo de forma a carregar dinamicamente o tokenizer/modelo adequado via `get_qa_pipeline(lang)`:
    * **Português:** BERT squad do Pierre Guillou (`pierreguillou/bert-base-cased-squad-v1.1-portuguese`).
    * **Inglês:** DistilBERT squad (`distilbert-base-cased-distilled-squad`).
  * O modelo devolve os índices e o texto da resposta, destacando-o a amarelo na interface (`templates/qa.html`) através da tag `<mark>`.
* **Restrição de Idioma (Enforcement):**
  * O sistema utiliza a função `detectar_idioma_pergunta(pergunta)` para verificar se o idioma da questão coincide com o do artigo. Caso não coincida (ex: pergunta em português num artigo em inglês), a pesquisa do BERT é abortada e uma mensagem de aviso vermelha é devolvida à interface.
* **Bónus - Pipeline de Fine-Tuning Completa:**
  * **Dataset de Ajuste:** Criado o ficheiro [dados_treino_qa.json](file:///c:/Users/Maria/Desktop/Universidade/mestrado/PLN/plneb-2526/Projetos/TP2/pln-tp2/dados_treino_qa.json) com um template em formato JSON contendo 5 exemplos anotados de perguntas e respostas médicas em português (com offsets de caracteres).
  * **Script de Treino:** Criado o script [treinar_qa.py](file:///c:/Users/Maria/Desktop/Universidade/mestrado/PLN/plneb-2526/Projetos/TP2/pln-tp2/treinar_qa.py) que lê os dados anotados, treina o BERT base com o `Trainer` da Hugging Face e gera um modelo local com pesos ajustados na pasta `./modelo_qa_pt_finetuned`.
  * **Deteção e Integração Automática:** A variável `_qa_pipelines['pt']` no `app.py` verifica automaticamente se a pasta local `./modelo_qa_pt_finetuned` existe. Em caso afirmativo, o servidor Flask carrega o vosso modelo local ajustado de forma completamente transparente.

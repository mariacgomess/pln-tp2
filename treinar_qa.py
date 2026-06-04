import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from torch.utils.data import Dataset

# 1. Carregar os Dados de Treino
try:
    with open("dados_treino_qa.json", "r", encoding="utf-8") as f:
        dados = json.load(f)
except FileNotFoundError:
    print("Erro: Ficheiro 'dados_treino_qa.json' não encontrado. Cria um primeiro!")
    exit(1)

model_name = "pierreguillou/bert-base-cased-squad-v1.1-portuguese"
print(f"A carregar tokenizer para '{model_name}'...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Dataset Customizado para PyTorch
class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=384):
        self.encodings = []
        for item in data:
            inputs = tokenizer(
                item["question"],
                item["context"],
                max_length=max_length,
                truncation="only_second",
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt"
            )
            
            offset_mapping = inputs["offset_mapping"][0]
            answer = item["answers"]
            start_char = answer["answer_start"]
            end_char = start_char + len(answer["text"])
            sequence_ids = inputs.sequence_ids()
            
            # Encontrar início do contexto nos tokens
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            
            # Encontrar fim do contexto nos tokens
            idx = len(sequence_ids) - 1
            while sequence_ids[idx] != 1:
                idx -= 1
            context_end = idx
            
            # Se a resposta estiver fora do contexto recortado
            if offset_mapping[context_start][0] > start_char or offset_mapping[context_end][1] < end_char:
                start_position = 0
                end_position = 0
            else:
                # Encontrar token de início
                token_start_idx = context_start
                while token_start_idx <= context_end and offset_mapping[token_start_idx][0] <= start_char:
                    token_start_idx += 1
                start_position = token_start_idx - 1
                
                # Encontrar token de fim
                token_end_idx = context_end
                while token_end_idx >= context_start and offset_mapping[token_end_idx][1] >= end_char:
                    token_end_idx -= 1
                end_position = token_end_idx + 1

            self.encodings.append({
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "start_positions": torch.tensor(start_position),
                "end_positions": torch.tensor(end_position)
            })

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]

print("A pré-processar o dataset...")
dataset = QADataset(dados, tokenizer)

# 3. Configurar e Treinar o Modelo
print(f"A carregar modelo base '{model_name}'...")
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./resultados_treino",
    num_train_epochs=3,              # Treinar por 3 épocas
    per_device_train_batch_size=4,   # Batch size pequeno para caber em qualquer CPU/GPU
    save_steps=50,
    logging_steps=10,
    learning_rate=3e-5,
    weight_decay=0.01,
    use_cpu=not torch.cuda.is_available() # Detetar se há GPU, se não usa CPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

print("\n--- A iniciar o Fine-Tuning do BERT ---")
trainer.train()

# 4. Guardar o modelo treinado localmente
print("\nA guardar modelo e tokenizer na pasta local './modelo_qa_pt_finetuned'...")
model.save_pretrained("./modelo_qa_pt_finetuned")
tokenizer.save_pretrained("./modelo_qa_pt_finetuned")
print("[OK] Sucesso! O modelo ajustado está pronto a ser usado pelo portal.")

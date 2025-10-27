import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import time

# Configuración de dispositivo
if torch.cuda.is_available():
    dispositivo = torch.device('cuda')
else:
    raise SystemExit("No se detectó GPU")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Clase Dataset
class DatasetReseñas(Dataset):
    def __init__(self, textos, etiquetas, tokenizer, longitud_maxima=64):
        self.textos = textos
        self.etiquetas = etiquetas
        self.tokenizer = tokenizer
        self.longitud_maxima = longitud_maxima
    
    def __len__(self):
        return len(self.textos)
    
    def __getitem__(self, idx):
        texto = str(self.textos[idx])
        etiqueta = self.etiquetas[idx]
        
        codificacion = self.tokenizer(
            texto,
            truncation=True,
            padding='max_length',
            max_length=self.longitud_maxima,
            return_tensors='pt'
        )
        
        return {
            'input_ids': codificacion['input_ids'].flatten(),
            'attention_mask': codificacion['attention_mask'].flatten(),
            'labels': torch.tensor(etiqueta, dtype=torch.long)
        }

# Cargar datos
def cargar_datos():
    X_entrenamiento = np.load('datasetDividido/X_train_text.npy', allow_pickle=True)
    X_validacion = np.load('datasetDividido/X_val_text.npy', allow_pickle=True)
    X_prueba = np.load('datasetDividido/X_test_text.npy', allow_pickle=True)

    y_entrenamiento = np.load('datasetDividido/y_train.npy', allow_pickle=True)
    y_validacion = np.load('datasetDividido/y_val.npy', allow_pickle=True)
    y_prueba = np.load('datasetDividido/y_test.npy', allow_pickle=True)

    MUESTRAS_ENTRENAMIENTO = 50000
    MUESTRAS_VALIDACION = 10000
    
    X_entrenamiento = X_entrenamiento[:MUESTRAS_ENTRENAMIENTO]
    y_entrenamiento = y_entrenamiento[:MUESTRAS_ENTRENAMIENTO]
    X_validacion = X_validacion[:MUESTRAS_VALIDACION]
    y_validacion = y_validacion[:MUESTRAS_VALIDACION]
    
    return (X_entrenamiento, X_validacion, X_prueba, 
            y_entrenamiento, y_validacion, y_prueba)

# Inicializar modelo
def inicializar_modelo():
    nombre_modelo = 'bert-base-uncased'

    tokenizer = BertTokenizer.from_pretrained(nombre_modelo)
    modelo = BertForSequenceClassification.from_pretrained(
        nombre_modelo,
        num_labels=2
    )
    
    modelo = modelo.to(dispositivo)
    return modelo, tokenizer

# Crear DataLoaders
def crear_dataloaders(X_entrenamiento, X_validacion, X_prueba, 
                     y_entrenamiento, y_validacion, y_prueba, tokenizer):
    batch_size = 32
    num_workers = 0
    longitud_maxima = 64

    dataset_entrenamiento = DatasetReseñas(X_entrenamiento, y_entrenamiento, tokenizer, longitud_maxima)
    dataset_validacion = DatasetReseñas(X_validacion, y_validacion, tokenizer, longitud_maxima)
    dataset_prueba = DatasetReseñas(X_prueba, y_prueba, tokenizer, longitud_maxima)

    cargador_entrenamiento = DataLoader(
        dataset_entrenamiento, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    cargador_validacion = DataLoader(
        dataset_validacion, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    cargador_prueba = DataLoader(
        dataset_prueba, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return cargador_entrenamiento, cargador_validacion, cargador_prueba

# Función de entrenamiento
def entrenar_epoca(modelo, cargador_datos, optimizador, dispositivo, epoca_num):
    modelo.train()
    perdida_total = 0
    batch_count = 0
    
    for lote in cargador_datos:
        optimizador.zero_grad()
        
        input_ids = lote['input_ids'].to(dispositivo, non_blocking=True)
        attention_mask = lote['attention_mask'].to(dispositivo, non_blocking=True)
        etiquetas = lote['labels'].to(dispositivo, non_blocking=True)
        
        salidas = modelo(input_ids, attention_mask=attention_mask, labels=etiquetas)
        perdida = salidas.loss
        
        perdida.backward()
        optimizador.step()
        
        perdida_total += perdida.item()
        batch_count += 1
        
        if batch_count % 100 == 0:
            porcentaje = (batch_count / len(cargador_datos)) * 100
            print(f"Epoca {epoca_num}: {batch_count}/{len(cargador_datos)} batches ({porcentaje:.1f}%)")
    
    return perdida_total / len(cargador_datos)

# Función de evaluación
def evaluar_modelo(modelo, cargador_datos, dispositivo):
    modelo.eval()
    predicciones = []
    etiquetas_reales = []
    
    with torch.no_grad():
        for lote in cargador_datos:
            input_ids = lote['input_ids'].to(dispositivo, non_blocking=True)
            attention_mask = lote['attention_mask'].to(dispositivo, non_blocking=True)
            etiquetas = lote['labels'].to(dispositivo, non_blocking=True)
            
            salidas = modelo(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(salidas.logits, dim=1)
            
            predicciones.extend(preds.cpu().numpy())
            etiquetas_reales.extend(etiquetas.cpu().numpy())
    
    exactitud = accuracy_score(etiquetas_reales, predicciones)
    return exactitud, predicciones, etiquetas_reales

# Función para predecir
def predecir_comentario(texto, modelo, tokenizer, dispositivo):
    modelo.eval()
    
    codificacion = tokenizer(
        texto,
        truncation=True,
        padding='max_length',
        max_length=64,
        return_tensors='pt'
    )
    
    input_ids = codificacion['input_ids'].to(dispositivo)
    attention_mask = codificacion['attention_mask'].to(dispositivo)
    
    with torch.no_grad():
        salidas = modelo(input_ids, attention_mask=attention_mask)
        probabilidades = torch.softmax(salidas.logits, dim=1)
        prediccion = torch.argmax(probabilidades, dim=1)
    
    clase = "RECOMENDABLE" if prediccion.item() == 1 else "NO RECOMENDABLE"
    confianza = probabilidades[0][prediccion.item()].item()
    
    return clase, confianza

# Función principal
def main():
    # Cargar datos
    (X_entrenamiento, X_validacion, X_prueba, 
     y_entrenamiento, y_validacion, y_prueba) = cargar_datos()
    
    # Inicializar modelo
    modelo, tokenizer = inicializar_modelo()
    
    # Crear dataloaders
    cargador_entrenamiento, cargador_validacion, cargador_prueba = crear_dataloaders(
        X_entrenamiento, X_validacion, X_prueba,
        y_entrenamiento, y_validacion, y_prueba, tokenizer
    )
    
    # Configurar entrenamiento
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=2e-5)

    # Entrenamiento
    epocas = 3
    perdidas_entrenamiento = []
    exactitudes_validacion = []

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    inicio_total = time.time()

    for epoca in range(epocas):
        print(f"Epoca {epoca + 1}/{epocas}")
        
        inicio_epoca = time.time()
        
        # Entrenamiento
        perdida_entrenamiento = entrenar_epoca(modelo, cargador_entrenamiento, optimizador, dispositivo, epoca + 1)
        perdidas_entrenamiento.append(perdida_entrenamiento)
        
        # Evaluación
        exactitud_validacion, _, _ = evaluar_modelo(modelo, cargador_validacion, dispositivo)
        exactitudes_validacion.append(exactitud_validacion)
        
        fin_epoca = time.time()
        tiempo_epoca = fin_epoca - inicio_epoca
        
        print(f"Perdida: {perdida_entrenamiento:.4f}")
        print(f"Exactitud validación: {exactitud_validacion:.4f}")
        print(f"Tiempo: {tiempo_epoca/60:.1f} minutos")
        print()

    # Evaluación final
    exactitud_prueba, predicciones_prueba, etiquetas_prueba = evaluar_modelo(modelo, cargador_prueba, dispositivo)
    print(f"Exactitud en prueba: {exactitud_prueba:.4f}")

    print("Reporte de Clasificación:")
    print(classification_report(etiquetas_prueba, predicciones_prueba, 
                              target_names=['No Recomendable', 'Recomendable']))

    # Guardar modelo
    modelo.save_pretrained('bert_clasificador_resenas')
    tokenizer.save_pretrained('bert_clasificador_resenas')
    print("Modelo guardado")

    # Pruebas
    test_comments = [
        "This product is excellent, I totally recommend it",
        "Very poor quality, does not work properly",
        "Good for the price, serves its purpose", 
        "Terrible experience, would never buy again",
        "Amazing quality, exceeded my expectations"
    ]

    for comment in test_comments:
        clase, confianza = predecir_comentario(comment, modelo, tokenizer, dispositivo)
        print(f"'{comment}'")
        print(f"{clase} ({confianza:.2%})")
        print()

    

    # Resumen final
    tiempo_total = (time.time() - inicio_total) / 3600
    print(f"Entrenamiento completado en {tiempo_total:.2f} horas")

if __name__ == '__main__':
    main()
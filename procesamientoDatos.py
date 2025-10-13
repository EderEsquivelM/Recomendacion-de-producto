import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os


# ============================================
# 1. CARGA DEL DATASET JSONL
# ============================================
data = []
with open("D:/PIA-IA/dataset/datasetFiltrado.jsonl", 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        try:
            review = json.loads(line)
            data.append(review)
            
            # Limitar para pruebas (opcional)
            if i >= 100000: 
                break
                
        except json.JSONDecodeError:
            print(f"Error en línea {i}")
            continue
        
        if i % 1000 == 0:
            print(f"Procesadas {i} reseñas...")

df = pd.DataFrame(data)
print(f"\nTotal de reseñas cargadas: {len(df)}")


# ============================================
# 2. CREACIÓN DE CAMPOS NUEVOS
# ============================================
print("\nCreando nuevos campos...")

# Combinar 'title' y 'text' en un solo campo
df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# Convertir a tipos numéricos
df['rating'] = df['rating'].astype(float)
df['helpful_vote'] = df['helpful_vote'].astype(int)

# Crear columna TARGET (1 = recomendable, 0 = no recomendable)
df['is_recommended'] = (df['rating'] >= 4).astype(int)


# ============================================
# 3. ANÁLISIS EXPLORATORIO
# ============================================
print("\nANÁLISIS EXPLORATORIO")
print(f"\nDistribución de ratings:")
print(df['rating'].value_counts().sort_index())

print(f"\nDistribución del target:")
print(df['is_recommended'].value_counts())
print(f"Proporción de recomendados: {df['is_recommended'].mean():.2%}")

print(f"\nEstadísticas de helpful_vote:")
print(df['helpful_vote'].describe())

print(f"\nLongitud promedio de texto:")
print(f"Caracteres: {df['combined_text'].str.len().mean():.0f}")
print(f"Palabras: {df['combined_text'].str.split().str.len().mean():.0f}")


# ============================================
# 4. BALANCEO (UNDERSAMPLING / AUN POR VERIFICAR) 
# ============================================
positive = df[df['is_recommended'] == 1]
negative = df[df['is_recommended'] == 0]

print(f"\nReseñas positivas: {len(positive)}")
print(f"Reseñas negativas: {len(negative)}")

# Si hay más del doble de positivas que negativas → preguntar si aplicar undersampling
if len(positive) > len(negative) * 2:
    opcion = int(input("\n¿Quieres aplicar balanceo (undersampling)?\n1.- Sí\n2.- No\nOpción: "))
    if opcion == 1:
        print("\nAplicando undersampling...")
        positive_sampled = positive.sample(n=len(negative)*2, random_state=42)
        df = pd.concat([positive_sampled, negative]).sample(frac=1, random_state=42)
        print(f"Nuevo tamaño balanceado: {len(df)}")
    else:
        print("\nSe omitió el balanceo de clases.")


# ============================================
# 5. DIVISIÓN DE DATOS (TRAIN / VAL / TEST)
# ============================================
print("\nDividiendo datos en train(Entrenamiento)/val(Validación)/test(Prueba)...")

# Separar texto, variables numéricas y etiquetas
texts = df['combined_text'].values
numeric_features = df[['rating', 'helpful_vote']].values
labels = df['is_recommended'].values

# 70% train, 15% val, 15% test
X_temp, X_test, y_temp, y_test, num_temp, num_test = train_test_split(
    texts, labels, numeric_features, 
    test_size=0.15, 
    random_state=42,
    stratify=labels)

X_train, X_val, y_train, y_val, num_train, num_val = train_test_split(
    X_temp, y_temp, num_temp,
    test_size=0.1765,  # 0.1765 * 0.85 ≈ 0.15 del total
    random_state=42,
    stratify=y_temp)

print(f"Train: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
print(f"Val:   {len(X_val)} ({len(X_val)/len(df)*100:.1f}%)")
print(f"Test:  {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")


# ============================================
# 6. NORMALIZAR FEATURES NUMÉRICAS (OPCIONAL)
# ============================================
opcion_normalizar = int(input("\n¿Deseas normalizar los datos numéricos?  \n1.Si\n2.No\n"))

if opcion_normalizar ==1:
    print("\n Aplicando normalización (StandardScaler)...")
    scaler = StandardScaler()
    num_train_scaled = scaler.fit_transform(num_train)
    num_val_scaled = scaler.transform(num_val)
    num_test_scaled = scaler.transform(num_test)
    print("Normalización completada correctamente.")
else:
    print("\nSe omitió la normalización de datos numéricos.")
    num_train_scaled = num_train
    num_val_scaled = num_val
    num_test_scaled = num_test
    scaler = None  # No se crea el objeto scaler


# ============================================
# 7. GUARDAR DATOS PROCESADOS
# ============================================
print("\nGuardando datos procesados...")

# Crear carpeta de salida si no existe
output_dir = "D:/PIA-IA/dataset/datasetDividido"
os.makedirs(output_dir, exist_ok=True)

# Guardar arrays numpy
np.save(f'{output_dir}/X_train_text.npy', X_train)
np.save(f'{output_dir}/X_val_text.npy', X_val)
np.save(f'{output_dir}/X_test_text.npy', X_test)

np.save(f'{output_dir}/X_train_numeric.npy', num_train_scaled)
np.save(f'{output_dir}/X_val_numeric.npy', num_val_scaled)
np.save(f'{output_dir}/X_test_numeric.npy', num_test_scaled)

np.save(f'{output_dir}/y_train.npy', y_train)
np.save(f'{output_dir}/y_val.npy', y_val)
np.save(f'{output_dir}/y_test.npy', y_test)

# Guardar scaler solo si se creó
if scaler is not None:
    with open(f"{output_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler guardado correctamente.")
else:
    print("No se guardó scaler porque no se aplicó normalización.")

# Guardar muestra para inspección
sample_df = df.head(10000)
sample_df.to_csv(f"{output_dir}/datasetPrueba.csv", index=False)

print("\nPreprocesamiento completado exitosamente!")

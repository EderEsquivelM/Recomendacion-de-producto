import json
import re

archivo_entrada = "D:/PIA-IA/dataset/Electronics.jsonl"
archivo_salida = "D:/PIA-IA/dataset/datasetFiltrado.jsonl"

# Contadores
total_datos = 0        
datos_invalidos = 0    
saltados_rating3 = 0   # Reseñas con rating == 3
guardados = 0          # Reseñas guardadas

with open(archivo_entrada, "r", encoding="utf-8") as fin, \
     open(archivo_salida, "w", encoding="utf-8") as fout:

    for i, dato in enumerate(fin, start=1):
        # Si la línea está vacía o solo tiene espacios, se ignora
        if not dato.strip():
            continue

        total_datos += 1

        # Intentar convertir la línea en JSON
        try:
            item = json.loads(dato)
        except json.JSONDecodeError:
            datos_invalidos += 1
            continue

        # Saltar reseñas con rating == 3 (neutras)
        if item.get("rating") == 3:
            saltados_rating3 += 1
            continue

        # Procesar campo 'helpful_vote'
        helpful = item.get("helpful_vote")

        if helpful is None:
            helpful = 0
        else:
            # Extraer número de una cadena con texto
            match = re.search(r'\d+', str(helpful))
            if match:
                helpful = int(match.group())
            else:
                helpful = 0

        # Crear un nuevo objeto con los campos importantes
        filtrado = {
            "rating": item.get("rating"),
            "title": item.get("title"),
            "text": item.get("text"),
            "helpful_vote": helpful
        }

        # Guardar el nuevo registro en el archivo de salida
        fout.write(json.dumps(filtrado, ensure_ascii=False) + "\n")
        guardados += 1

print("Resumen del proceso:")
print(f"   Total de líneas leídas: {total_datos}")
print(f"   Líneas inválidas (JSON mal formado): {datos_invalidos}")
print(f"   Saltadas por rating == 3: {saltados_rating3}")
print(f"   Guardadas en {archivo_salida}: {guardados}")

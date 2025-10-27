import json
import random

archivo_entrada = "Electronics.jsonl"
archivo_salida = "datasetFiltrado.jsonl"
limite = 1000000

total_datos = 0
datos_invalidos = 0
saltados_rating3 = 0
reservorio = []

print("Procesando con muestreo aleatorio...")

with open(archivo_entrada, "r", encoding="utf-8") as fin:
    for linea in fin:
        if not linea.strip():
            continue

        total_datos += 1

        try:
            item = json.loads(linea)
        except json.JSONDecodeError:
            datos_invalidos += 1
            continue

        if item.get("rating") == 3:
            saltados_rating3 += 1
            continue

        filtrado = {
            "rating": item.get("rating"),
            "title": item.get("title"),
            "text": item.get("text"),
        }

        # Reservoir Sampling
        if len(reservorio) < limite:
            reservorio.append(filtrado)
        else:
            j = random.randint(0, total_datos - 1)
            if j < limite:
                reservorio[j] = filtrado

print(f"\nGuardando {len(reservorio):,} reseñas aleatorias...")

with open(archivo_salida, "w", encoding="utf-8") as fout:
    for item in reservorio:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\nProceso completado")
print(f"   Total líneas leídas: {total_datos:,}")
print(f"   JSON inválidos: {datos_invalidos:,}")
print(f"   Saltados rating == 3: {saltados_rating3:,}")
print(f"   Guardados en {archivo_salida}: {len(reservorio):,}")

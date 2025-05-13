import os
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Inicializar cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cargar CSVs desde la carpeta /csvs
csv_folder = os.path.join(os.path.dirname(__file__), "csvs")
csv_files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith(".csv")]
df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# Inicializar FastAPI
app = FastAPI()

# Configurar CORS para permitir peticiones desde cualquier origen (ajustable)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prompt del sistema
system_prompt_csv = """
Sos un asistente que responde preguntas sobre los gastos del municipio de Bahía Blanca (2008–2023).
Tenés acceso a una tabla con las siguientes columnas: año, orden de compra, fecha, importe, proveedor, dependencia y expediente.

Si el usuario hace una pregunta concreta, respondé en este formato:

Filtro: [condición en pandas (por ejemplo: año == 2022 and dependencia == "Salud")]
Resumen: [explicación que acompañe la respuesta]

NO inventes datos. Solo indicá cómo filtrar.
"""

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_message = body.get("message")
    print(f"📩 Mensaje recibido: {user_message}")

    try:
        # Solicitar al modelo que genere el filtro
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt_csv},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
        )

        response_text = completion.choices[0].message.content.strip()
        print(f"🧠 Respuesta del modelo:\n{response_text}")

        if "Filtro:" in response_text:
            filtro = response_text.split("Filtro:")[1].split("Resumen:")[0].strip()
            resumen = response_text.split("Resumen:")[1].strip()

            print(f"📎 Filtro generado: {filtro}")

            # Aplicar el filtro al DataFrame
            resultados = df.query(filtro)

            # Limpiar datos no serializables
            filas = (
                resultados.head(10)
                .replace({float("inf"): None, float("-inf"): None})
                .fillna("N/A")
                .to_dict(orient="records")
            )

            return {
                "message": resumen,
                "results": filas,
                "filter": filtro
            }

        # Si no hay filtro, devolver solo el texto del modelo
        return {
            "message": response_text,
            "results": [],
            "filter": None
        }

    except Exception as e:
        print("❌ Error en el backend:", str(e))
        return {
            "error": f"Ocurrió un error al procesar la solicitud: {str(e)}"
        }

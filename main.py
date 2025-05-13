import os
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Cliente OpenAI (nuevo SDK)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Carga de CSVs
csv_folder = os.path.join(os.path.dirname(__file__), "csvs")
csv_files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith(".csv")]
df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# FastAPI
app = FastAPI()

# CORS
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

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt_csv},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
        )
        response_text = completion.choices[0].message.content.strip()
        print("🧠 Respuesta del modelo:", response_text)

        if "Filtro:" in response_text:
            filtro = response_text.split("Filtro:")[1].split("Resumen:")[0].strip()
            resumen = response_text.split("Resumen:")[1].strip()

            resultados = df.query(filtro)
            filas = resultados.head(10).to_dict(orient="records")

            return {
                "message": resumen,
                "results": filas,
                "filter": filtro
            }

        return {"message": response_text}

    except Exception as e:
        print("❌ Error:", str(e))
        return {
            "error": f"Ocurrió un error al procesar la solicitud: {str(e)}"
        }

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import glob
import os
import numpy as np

app = FastAPI(title="Plume Q-Learning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def calcular_media_movel(dados, janela=50):
    if len(dados) < janela:
        return dados
    return (np.convolve(dados, np.ones(janela), "valid") / janela).tolist()


DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
# Define a pasta alvo
PASTA_MODELOS = os.path.normpath(os.path.join(DIRETORIO_ATUAL, "..", "modelos"))


@app.get("/api/modelos")
def listar_modelos():
    if not os.path.exists(PASTA_MODELOS):
        return {"modelos": []}

    modelos = glob.glob(os.path.join(PASTA_MODELOS, "qlearning_plume_*.pkl"))
    modelos.sort(key=os.path.getmtime, reverse=True)

    # Enviar para o React apenas os nomes dos ficheiros, para o dropdown ficar limpo
    nomes_modelos = [os.path.basename(m) for m in modelos]
    return {"modelos": nomes_modelos}


@app.get("/api/dados/{nome_modelo}")
def obter_dados(nome_modelo: str):
    # Reconstruir o caminho para abrir o ficheiro
    caminho_completo = os.path.join(PASTA_MODELOS, nome_modelo)

    if not os.path.exists(caminho_completo):
        return {"erro": "Ficheiro não encontrado"}

    with open(caminho_completo, "rb") as f:
        data = pickle.load(f)

    if "stats" not in data:
        return {"erro": "Este ficheiro não contém estatísticas guardadas."}

    stats = data["stats"]
    recompensas = stats["rewards"]
    passos = stats["lengths"]
    sucessos = stats["successes"]

    rec_ma = calcular_media_movel(recompensas, 50)
    pas_ma = calcular_media_movel(passos, 50)
    suc_ma = [s * 100 for s in calcular_media_movel(sucessos, 50)]

    return {
        "sucesso": True,
        "episodios": list(range(1, len(recompensas) + 1)),
        "recompensas": recompensas,
        "recompensas_ma": rec_ma,
        "passos": passos,
        "passos_ma": pas_ma,
        "sucessos_ma": suc_ma,
        "tamanho_qtable": len(data.get("q_table", {})),
        "epsilon_final": round(data.get("epsilon", 0), 4),
    }


if __name__ == "__main__":
    import uvicorn

    print("\n🚀 INICIAR API FASTAPI EM: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

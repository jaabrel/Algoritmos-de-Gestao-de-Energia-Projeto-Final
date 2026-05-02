import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def plot_training_stats(filename):
    if not os.path.exists(filename):
        print(f"Erro: O ficheiro '{filename}' não existe.")
        return
    with open(filename, "rb") as f:
        data = pickle.load(f)

    if "stats" not in data:
        print("Erro: Este ficheiro não contém estatísticas de treino antigas.")
        return
    stats = data["stats"]
    rewards = stats["rewards"]
    lengths = stats["lengths"]
    successes = stats["successes"]
    episodes = range(1, len(rewards) + 1)

    # Função auxiliar para calcular a média móvel (para suavizar os gráficos)
    def moving_average(x, w=50):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w), "valid") / w

    # Criar a figura com 3 gráficos
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f"Análise de Treino: {filename}", fontsize=16)

    # Gráfico 1: Recompensas
    axs[0].plot(
        episodes, rewards, alpha=0.3, color="blue", label="Recompensa por Episódio"
    )
    axs[0].plot(
        range(50, len(rewards) + 1),
        moving_average(rewards, 50),
        color="darkblue",
        label="Média Móvel (50)",
    )
    axs[0].set_ylabel("Recompensa")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Gráfico 2: Duração do Episódio (Passos)
    axs[1].plot(
        episodes, lengths, alpha=0.3, color="orange", label="Passos por Episódio"
    )
    axs[1].plot(
        range(50, len(lengths) + 1),
        moving_average(lengths, 50),
        color="darkorange",
        label="Média Móvel (50)",
    )
    axs[1].set_ylabel("Nº de Passos")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # Gráfico 3: Taxa de Sucesso (%)
    # Convertemos para percentagem numa janela rolante
    success_rate = moving_average(successes, 50) * 100
    axs[2].plot(
        range(50, len(successes) + 1),
        success_rate,
        color="green",
        label="Taxa de Sucesso (Média 50 eps)",
    )
    axs[2].set_xlabel("Episódio")
    axs[2].set_ylabel("Sucesso (%)")
    axs[2].set_ylim([-5, 105])
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Permite passar o ficheiro via terminal: python analisador.py qlearning_plume_20231024_1530.pkl
    # Se não passar argumento, lista os disponíveis e pede para escolher
    import glob

    modelos = glob.glob("qlearning_plume_*.pkl")

    if len(sys.argv) > 1:
        plot_training_stats(sys.argv[1])
    elif modelos:
        print("Modelos disponíveis:")
        for i, m in enumerate(modelos):
            print(f" {i + 1}. {m}")
        escolha = int(input("\nQual modelo queres analisar? (Número): ")) - 1
        if 0 <= escolha < len(modelos):
            plot_training_stats(modelos[escolha])
        else:
            print("Escolha inválida.")
    else:
        print("Nenhum modelo encontrado na pasta.")

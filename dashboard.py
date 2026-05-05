import os
import glob
import pickle
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dashboard Q-Learning")
        self.root.geometry("900x800")

        self.pasta_modelos = "modelos"

        # --- PAINEL SUPERIOR (Controlos) ---
        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(
            control_frame, text="Selecione o Modelo:", font=("Arial", 11, "bold")
        ).pack(side=tk.LEFT, padx=5)

        self.combo_modelos = ttk.Combobox(
            control_frame, state="readonly", width=45, font=("Arial", 10)
        )
        self.combo_modelos.pack(side=tk.LEFT, padx=5)
        self.combo_modelos.bind("<<ComboboxSelected>>", self.carregar_dados)

        self.btn_atualizar = ttk.Button(
            control_frame, text="Atualizar Lista", command=self.listar_modelos
        )
        self.btn_atualizar.pack(side=tk.LEFT, padx=10)

        # --- PAINEL DE ESTATÍSTICAS ---
        stats_frame = ttk.Frame(root, padding="5")
        stats_frame.pack(side=tk.TOP, fill=tk.X)

        self.lbl_stats = ttk.Label(
            stats_frame,
            text="A carregar...",
            font=("Consolas", 11),
            foreground="#333333",
        )
        self.lbl_stats.pack()

        # --- PAINEL DOS GRÁFICOS (Canvas) ---
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        self.canvas_widget = None
        self.listar_modelos()

    def listar_modelos(self):
        """Procura todos os ficheiros .pkl e atualiza o dropdown."""
        if not os.path.exists(self.pasta_modelos):
            messagebox.showerror(
                "Erro",
                f"Pasta '{self.pasta_modelos}' não encontrada. Treina o agente primeiro.",
            )
            return

        modelos = glob.glob(os.path.join(self.pasta_modelos, "qlearning_plume_*.pkl"))
        modelos.sort(key=os.path.getmtime, reverse=True)
        self.modelos_dict = {os.path.basename(m): m for m in modelos}

        if self.modelos_dict:
            self.combo_modelos["values"] = list(self.modelos_dict.keys())
            self.combo_modelos.current(0)  # Seleciona o mais recente por defeito
            self.carregar_dados()
        else:
            self.combo_modelos.set("")
            self.lbl_stats.config(
                text="Nenhum modelo encontrado. Inicia um treino primeiro."
            )
            if self.canvas_widget:
                self.canvas_widget.get_tk_widget().destroy()

    def carregar_dados(self, event=None):
        """Carrega os dados do ficheiro selecionado e atualiza a interface."""
        selecao = self.combo_modelos.get()
        if not selecao:
            return

        caminho_completo = self.modelos_dict[selecao]

        try:
            with open(caminho_completo, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            messagebox.showerror(
                "Erro ao carregar", f"Não foi possível ler o ficheiro:\n{e}"
            )
            return

        stats = data.get("stats", {})
        if not stats or "rewards" not in stats:
            messagebox.showwarning(
                "Aviso", "Este ficheiro não contém um histórico de treino válido."
            )
            return

        # Calcular estatísticas finais
        sucessos = stats["successes"]
        tamanhos = stats["lengths"]
        ultimos_100_sucesso = (
            np.mean(sucessos[-100:]) * 100
            if len(sucessos) >= 100
            else np.mean(sucessos) * 100
        )
        ultimos_100_passos = (
            np.mean(tamanhos[-100:]) if len(tamanhos) >= 100 else np.mean(tamanhos)
        )

        texto_stats = (
            f"Episódios Totais: {len(stats['rewards'])}   |   "
            f"Tabela Q: {len(data.get('q_table', {}))} estados   |   "
            f"Sucesso Final: {ultimos_100_sucesso:.1f}%   |   "
            f"Passos (Média): {ultimos_100_passos:.1f}   |   "
            f"Exploração Final: {data.get('epsilon', 0.0):.4f}"
        )
        self.lbl_stats.config(text=texto_stats)

        # Passa o dicionário 'stats' inteiro para os gráficos
        self.desenhar_graficos(stats)

    def desenhar_graficos(self, stats):
        """Embutir os gráficos Matplotlib na interface Tkinter."""
        recompensas, tamanhos, sucessos = (
            stats["rewards"],
            stats["lengths"],
            stats["successes"],
        )
        paths = stats.get("paths", {})

        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()

        # Criar uma grelha 2x2 para acomodar o mapa
        fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
        window = max(1, len(recompensas) // 50)

        # 1. Recompensas (Topo Esquerda)
        axs[0, 0].plot(recompensas, alpha=0.3, color="royalblue")
        axs[0, 0].plot(
            np.convolve(recompensas, np.ones(window) / window, mode="valid"),
            color="navy",
            linewidth=2,
        )
        axs[0, 0].set_title("Recompensas Acumuladas", fontsize=10)
        axs[0, 0].grid(True, alpha=0.3)

        # 2. MAPA DE ROTAS (Topo Direita)
        axs[0, 1].set_title("Mapa de Navegação (Últimos Caminhos)", fontsize=10)
        axs[0, 1].set_xlim(0, 1)  # As coordenadas gravadas estão normalizadas [0, 1]
        axs[0, 1].set_ylim(0, 1)

        if paths:
            eps_gravados = sorted(paths.keys())

            # Desenha uma rota inicial (para ver como ele era no início)
            if eps_gravados:
                p_early = np.array(paths[eps_gravados[0]])
                axs[0, 1].plot(
                    p_early[:, 0],
                    p_early[:, 1],
                    color="red",
                    alpha=0.3,
                    label="Início do Treino",
                )

            # Desenha as últimas 5 rotas (para ver o comportamento ótimo)
            for ep in eps_gravados[-5:]:
                p = np.array(paths[ep])
                axs[0, 1].plot(p[:, 0], p[:, 1], color="forestgreen", alpha=0.6)

            # Adiciona uma legenda genérica para a cor verde
            axs[0, 1].plot(
                [], [], color="forestgreen", label="Treino Final (Últimos Eps)"
            )
            axs[0, 1].legend(loc="lower left", fontsize=8)
        else:
            axs[0, 1].text(
                0.5,
                0.5,
                "Sem dados de GPS.\nTreina um modelo novo!",
                ha="center",
                va="center",
                color="gray",
            )
        axs[0, 1].grid(True, alpha=0.3)

        # 3. Passos por Episódio (Baixo Esquerda)
        axs[1, 0].plot(tamanhos, alpha=0.3, color="darkorange")
        axs[1, 0].plot(
            np.convolve(tamanhos, np.ones(window) / window, mode="valid"),
            color="saddlebrown",
            linewidth=2,
        )
        axs[1, 0].set_title("Duração da Busca (Passos)", fontsize=10)
        axs[1, 0].grid(True, alpha=0.3)

        # 4. Taxa de Sucesso (Baixo Direita)
        sucesso_acumulado = np.cumsum(sucessos) / np.arange(1, len(sucessos) + 1) * 100
        axs[1, 1].plot(sucesso_acumulado, color="forestgreen", linewidth=2)
        axs[1, 1].set_title("Taxa de Sucesso Acumulada (%)", fontsize=10)
        axs[1, 1].set_xlabel("Episódio", fontsize=10)
        axs[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Ligar o Matplotlib ao Tkinter
        self.canvas_widget = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    # Configurar estilo para parecer mais moderno que o Tkinter normal
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")  # Temas comuns: 'clam', 'alt', 'default', 'classic'

    app = DashboardApp(root)
    root.mainloop()

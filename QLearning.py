"""
Módulo de Aprendizagem por Reforço (Q-Learning) para Navegação em Plumas de Odor.

Este módulo implementa o agente QLearningAgent, responsável por aprender
a política ótima de navegação do veículo autónomo (AUV) para localizar a fonte
de emissão da pluma através de amostragem sensorial e discretização de estados.
"""

import numpy as np
import random
import pickle
import datetime
import os
from collections import defaultdict


class QLearningAgent:
    """
    Agente de Aprendizagem por Reforço baseado no algoritmo Q-Learning tabular.
    
    O espaço de estados contínuo (posição, concentração de odor, direção da corrente,
    estado da missão e nível de bateria) é discretizado em intervalos finitos (bins).
    """

    def __init__(
        self,
        env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.998,
        min_exploration_rate: float = 0.01,
        state_bins: dict = None,
    ):
        """
        Inicializa o agente Q-Learning com os respetivos hiperparâmetros.

        Argumentos:
            env: Ambiente de simulação compatível com Gymnasium (PlumeRLWrapper).
            learning_rate: Taxa de aprendizagem (alpha), controla o peso das novas informações.
            discount_factor: Fator de desconto temporal (gamma), pondera recompensas futuras.
            exploration_rate: Taxa inicial de exploração estocástica (epsilon).
            exploration_decay: Fator multiplicativo de decaimento do epsilon por episódio.
            min_exploration_rate: Valor mínimo admissível para a taxa de exploração.
            state_bins: Dicionário com o número de partições discretas para cada dimensão.
        """
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration_rate

        # Configuração padrão de partições para discretização do estado contínuo
        self.state_bins = state_bins or {
            "x": 20,       # Partições ao longo do comprimento (Eixo X)
            "y": 10,       # Partições ao longo da largura (Eixo Y)
            "conc": 3,     # Níveis de concentração de odor (baixa, média, alta)
            "wind": 4,     # Quadrantes da direção do escoamento/vento (4 pontos cardeais)
            "battery": 3,  # Níveis de carga da bateria (baixa, média, alta)
        }

        # Tabela Q: mapeia tuplos de estado discreto para arrays de valores Q por ação
        self.q_table = {}

        # Histórico de métricas de treino para análise estatística
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_history = []
        self.saved_paths = {}

    def _state_key(self, state) -> tuple:
        """
        Converte o vetor contínuo de observação num tuplo discreto para indexação na Q-Table.

        Estrutura do estado:
            [norm_x, norm_y, conc, flow_x, flow_y, mission_state, battery_level]
        Todos os valores encontram-se normalizados no intervalo [0, 1].
        """
        state = np.asarray(state, dtype=np.float32)

        # 1. Discretização espacial (Posição X e Y)
        x_b = int(np.clip(state[0] * self.state_bins["x"], 0, self.state_bins["x"] - 1))
        y_b = int(np.clip(state[1] * self.state_bins["y"], 0, self.state_bins["y"] - 1))

        # 2. Discretização da concentração de odor recebida no sensor
        conc = state[2]
        if conc <= 0.33:
            c_b = 0  # Concentração nula ou muito baixa
        elif conc <= 0.67:
            c_b = 1  # Concentração moderada (limiar da pluma)
        else:
            c_b = 2  # Concentração elevada (núcleo da pluma)

        # 3. Discretização da orientação do escoamento do fluido (4 quadrantes)
        fx, fy = state[3], state[4]
        angle = np.arctan2(fy, fx)
        wind_dir = int(np.round(angle / (np.pi / 2))) % 4

        # Compatibilidade retroativa com modelos anteriores (tuplos de 4 ou 5 elementos)
        if self.q_table:
            first_key = next(iter(self.q_table.keys()))
            if len(first_key) == 4:
                return (x_b, y_b, c_b, wind_dir)
            elif len(first_key) == 5:
                mission_state_bin = 0
                if len(state) > 5:
                    mission_state_bin = int(np.round(state[5]))
                return (x_b, y_b, c_b, wind_dir, mission_state_bin)

        # 4. Estado da missão (exploração, seguimento de fronteira, retorno)
        mission_state_bin = 0
        if len(state) > 5:
            mission_state_bin = int(np.round(state[5]))

        # 5. Nível de carga da bateria (crítico, operacional, elevado)
        battery_bin = 0
        if len(state) > 6:
            battery_val = state[6]
            if battery_val <= 0.3:
                battery_bin = 0  # Bateria baixa (<30%)
            elif battery_val <= 0.7:
                battery_bin = 1  # Bateria média
            else:
                battery_bin = 2  # Bateria alta

        return (x_b, y_b, c_b, wind_dir, mission_state_bin, battery_bin)

    def choose_action(self, state, greedy: bool = False) -> int:
        """
        Seleciona a ação a executar segundo a política epsilon-greedy.

        Argumentos:
            state: Vetor de estado atual retornado pelo ambiente.
            greedy: Se True, força a seleção determinística da melhor ação conhecida.

        Retorna:
            Índice da ação selecionada (0: Cima, 1: Baixo, 2: Esquerda, 3: Direita).
        """
        key = self._state_key(state)

        # Fase de Exploração: escolha estocástica aleatória
        if not greedy and random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            # Fase de Aproveitamento (Exploitation): melhor ação aprendida
            q_values = self.q_table.get(key, np.zeros(self.env.action_space.n))
            if np.all(q_values == 0):
                # Estado novo sem histórico prévio: sorteia ação para evitar bloqueios
                return random.randint(0, self.env.action_space.n - 1)
            return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, done, truncated=False):
        """
        Atualiza o valor Q na tabela através da Equação de Bellman para Q-Learning:
        Q(s, a) = Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
        """
        key = self._state_key(state)
        next_key = self._state_key(next_state)

        # Inicializa o estado com zeros caso seja a primeira visita
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.env.action_space.n)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.env.action_space.n)

        current_q = self.q_table[key][action]

        # Tratamento do estado terminal (sem estimativa de recompensas futuras)
        if done or truncated:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_key])
            target_q = reward + self.gamma * max_next_q

        # Aplicação da taxa de aprendizagem à discrepância temporal
        self.q_table[key][action] += self.lr * (target_q - current_q)

    def train(self, env, num_episodes=500, print_every=50):
        """
        Executa o ciclo completo de treino do agente com registo de telemetria e rotas.

        Argumentos:
            env: Instância do ambiente PlumeRLWrapper.
            num_episodes: Quantidade de episódios a treinar.
            print_every: Intervalo de episódios para apresentação do progresso no terminal.
        """
        import logging
        pasta_logs = "logs"
        os.makedirs(pasta_logs, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(pasta_logs, f"treino_{timestamp}.log")

        logger = logging.getLogger(f"QLearningTrain_{timestamp}")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(fh)

        logger.info("=== INÍCIO DO TREINO Q-LEARNING ===")
        logger.info(f"Episódios a treinar: {num_episodes}")
        logger.info(f"Taxa de Aprendizagem (Alpha): {self.lr}, Fator de Desconto (Gamma): {self.gamma}")
        logger.info(f"Epsilon Inicial: {self.epsilon}, Decaimento: {self.epsilon_decay}, Mínimo: {self.min_epsilon}")
        logger.info(f"Partições de Estado (Bins): {self.state_bins}")

        print(f"\n[INFO] Ficheiro de registo detalhado do treino criado em: {log_filename}")
        print(f"\n{'=' * 70}")
        print(f"  TREINO DO AGENTE Q-LEARNING — {num_episodes} episódios")
        print(
            f"  Partições de estado: {self.state_bins['x']}×{self.state_bins['y']}×{self.state_bins['conc']}×{self.state_bins['wind']}"
        )
        print(f"{'=' * 70}\n")

        episodios_iniciais = len(self.episode_rewards)

        for ep_relativo in range(1, num_episodes + 1):
            ep_absoluto = episodios_iniciais + ep_relativo

            state, info = env.reset()
            current_path = [(state[0], state[1])]
            done = truncated = False
            total_reward = 0
            step = 0

            # Execução dos passos do episódio
            while not (done or truncated):
                action = self.choose_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                current_path.append((next_state[0], next_state[1]))

                self.update(state, action, reward, next_state, done, truncated)

                state = next_state
                total_reward += reward
                step += 1

            # Registo de estatísticas
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(step)
            is_success = info.get("at_goal", False)
            self.success_history.append(1 if is_success else 0)

            # Registo no ficheiro de log
            logger.info(
                f"Episódio {ep_absoluto} | Recompensa: {total_reward:.2f} | "
                f"Passos: {step} | Sucesso: {is_success} | "
                f"Epsilon: {self.epsilon:.4f} | Estados Q: {len(self.q_table)}"
            )

            # Decaimento progressivo da taxa de exploração estocástica
            if self.epsilon > self.min_epsilon:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Paragem antecipada (Early Stopping) se a taxa de sucesso estabilizar nos 80%
            if ep_relativo >= 100:
                w_check = min(100, len(self.success_history))
                avg_success_check = np.mean(self.success_history[-w_check:])
                if avg_success_check >= 0.80:
                    logger.info(f"Convergência atingida com {avg_success_check*100:.1f}% de sucesso no ep {ep_absoluto}. Paragem antecipada.")
                    print(f"\n[Early Stopping] Convergência verificada ({avg_success_check*100:.1f}% de sucesso) no episódio {ep_absoluto}!")
                    break

            # Armazenamento seletivo de trajetórias para inspeção no mapa do Dashboard
            save_interval = 1 if num_episodes <= 100 else (5 if num_episodes <= 500 else 20)
            if ep_relativo <= 10 or ep_relativo > num_episodes - 25 or ep_absoluto % save_interval == 0:
                self.saved_paths[ep_absoluto] = current_path

            # Apresentação do progresso periódico na consola
            if ep_relativo % print_every == 0:
                w = min(print_every, ep_relativo)
                avg_reward = np.mean(self.episode_rewards[-w:])
                avg_length = np.mean(self.episode_lengths[-w:])
                avg_success = np.mean(self.success_history[-w:]) * 100

                print(
                    f"  Ep {ep_absoluto:>5} (+{ep_relativo}) | "
                    f"Recompensa Média: {avg_reward:>7.2f} | "
                    f"Passos Médios: {avg_length:>6.1f} | "
                    f"Sucesso: {avg_success:>5.1f}% | "
                    f"Eps: {self.epsilon:.4f} | "
                    f"Estados Q: {len(self.q_table):>5}"
                )

        logger.info("=== CONCLUSÃO DO TREINO ===")
        logger.removeHandler(fh)
        fh.close()

    def save(self, filename=None):
        """
        Serializa e guarda o modelo treinado, a Q-Table e as métricas num ficheiro .pkl.

        Argumentos:
            filename: Nome opcional para o ficheiro (se None, gera carimbo de data/hora).

        Retorna:
            Caminho absoluto ou relativo do ficheiro gravado.
        """
        pasta = "modelos"
        os.makedirs(pasta, exist_ok=True)

        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qlearning_plume_{timestamp}.pkl"
        elif not filename.endswith(".pkl"):
            filename = f"{filename}.pkl"

        caminho_completo = os.path.join(pasta, filename)

        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "state_bins": self.state_bins,
            "hyperparameters": {
                "learning_rate": self.lr,
                "discount_factor": self.gamma,
                "exploration_decay": self.epsilon_decay,
                "min_epsilon": self.min_epsilon,
            },
            "stats": {
                "rewards": self.episode_rewards,
                "lengths": self.episode_lengths,
                "successes": self.success_history,
                "paths": getattr(self, "saved_paths", {}),
            },
            "source_pos": getattr(self.env, 'config', {}).get("source_position", (20.0, 25.0)),
            "world_size": (getattr(self.env, 'config', {}).get("world_width", 100.0), getattr(self.env, 'config', {}).get("world_height", 50.0))
        }

        with open(caminho_completo, "wb") as f:
            pickle.dump(data, f)

        print(f"\n[+] Modelo e estatísticas gravados com sucesso em: {caminho_completo}")
        print(f"    Episódios acumulados: {len(self.episode_rewards)}")
        print(f"    Estados descobertos na Q-Table: {len(self.q_table)}")
        print(
            f"    Taxa final de sucesso: {np.mean(self.success_history[-100:]) * 100:.1f}%"
        )

        return caminho_completo

    def load(self, filepath):
        """
        Carrega um modelo pré-treinado a partir de um ficheiro .pkl.

        Argumentos:
            filepath: Caminho para o ficheiro do modelo a carregar.

        Retorna:
            True em caso de sucesso, False se ocorrer algum erro ou o ficheiro não existir.
        """
        if not os.path.exists(filepath):
            print(f"[-] AVISO: Ficheiro {filepath} não encontrado.")
            return False

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            self.q_table.update(data["q_table"])
            self.epsilon = data.get("epsilon", self.epsilon)
            self.state_bins = data.get("state_bins", self.state_bins)

            if "stats" in data:
                self.episode_rewards = data["stats"].get("rewards", [])
                self.episode_lengths = data["stats"].get("lengths", [])
                self.success_history = data["stats"].get("successes", [])
                self.saved_paths = data["stats"].get("paths", {})

            print(f"[+] Modelo carregado com sucesso de {filepath}")
            print(f"    Estados na Q-Table: {len(self.q_table)}")
            print(f"    Histórico de episódios: {len(self.episode_rewards)}")

            return True

        except Exception as e:
            print(f"[-] Erro ao carregar o modelo: {e}")
            return False

    def get_statistics(self):
        """Retorna um resumo com as principais métricas de desempenho do treino."""
        if not self.episode_rewards:
            return None

        return {
            "total_episodes": len(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards),
            "avg_length": np.mean(self.episode_lengths),
            "success_rate": np.mean(self.success_history),
            "q_table_size": len(self.q_table),
            "current_epsilon": self.epsilon,
        }

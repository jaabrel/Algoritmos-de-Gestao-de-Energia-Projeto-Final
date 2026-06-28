# Explicação das Alterações ao Código

Este documento detalha as modificações efetuadas nos ficheiros do projeto para implementar a gestão de bateria, o movimento de busca em ziguezague (casting) do agente e a nova física da pluma.

---

## 1. Funcionalidade da Bateria

A bateria é consumida continuamente durante o movimento e sofre maior decaimento quando o agente se desloca contra o vento devido à resistência.

### Ficheiro: [config.json](file:///C:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/config.json)
* **Parâmetros da Bateria (Linhas 44-47)**:
  * `"battery_capacity": 100.0` (Capacidade total calibrada para base 100).
  * `"battery_base_decay": 0.5` (Decaimento base por segundo).
  * `"battery_wind_factor": 1.5` (Fator multiplicador da resistência do vento).
  * `"battery_min_decay": 0.1` (Decaimento mínimo garantido).

### Ficheiro: [main.py](file:///C:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/main.py)
* **Inicialização no Agente (Linhas 352-355 e 376)**:
  * Configura a capacidade e decaimento base no `__init__` da classe `Agent` e reinicia `self.battery = self.battery_capacity` no `reset()`.
* **Cálculo de Consumo no Treino / Wrappers (Linhas 764-783)**:
  * Na classe `PlumeRLWrapper.step()`, calcula a resistência do vento (`np.dot(direction, opposing_wind)`) e atualiza o estado da bateria. O episódio termina se a bateria esgotar (`self.battery <= 0`).
* **Cálculo de Consumo no Simulador Real-Time (Linhas 1311-1325)**:
  * No método `Simulator.step()`, desconta a energia consumida no passo com base no deslocamento real efetuado pelo agente (`displacement`) e na resistência do vento em tempo real.
* **Visualização da Bateria (Linha 1610)**:
  * Adicionado ao HUD da simulação (`HUD Info`) no ecrã OpenCV: `Bat: {self.agent.battery:.1f}/{self.agent.battery_capacity:.1f}`.

---

## 2. Comportamento de Busca em Ziguezague (Casting)

Garante que o agente, enquanto está no estado de busca inicial (`"searching"`), faça um padrão regular de ziguezague subindo e descendo crosswind enquanto avança contra o vento (upwind).

### Ficheiro: [main.py](file:///C:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/main.py)
* **Inicialização das Variáveis de Controlo (Linhas 1023-1026)**:
  * No método `QLearningSimAgent.reset()`, inicializamos `self.search_zig_zag_dir` (que alterna entre `1` e `-1`) e `self.search_steps_in_dir = 0.0` para contar a distância percorrida na diagonal atual.
* **Algoritmo de Movimento em Ziguezague (Linhas 1029-1079)**:
  * No método `QLearningSimAgent.run_search_algorithm()`, quando `self.state == "searching"`, o agente:
    1. Obtém a direção contra o vento (`upwind_dir = -self.measured_flow_direction / flow_norm`).
    2. Rotaciona esta direção diagonalmente pelo ângulo de cast configurado (`cast_angle`, tipicamente 60º) na direção atual (`search_zig_zag_dir`).
    3. Atualiza a posição calculando o passo diagonal.
    4. Alterna de direção (`self.search_zig_zag_dir *= -1` e zera a distância) ao atingir o limite de cast (`cast_length`, ex: 6.0 unidades) ou se aproximar das paredes do topo/rodapé do mapa (margem de 2.0 unidades).

---

## 3. Física e Comportamento da Pluma

A pluma foi redesenhada para ser mais larga (grossa) e fácil de intercetar, mas sem perder a organização das partículas (reduzindo a turbulência caótica e confiando no meandro amplo).

### Ficheiro: [config.json](file:///C:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/config.json) e [main.py](file:///C:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/main.py)
* **Intensidade da Turbulência (config.json linha 38 / main.py linha 1682)**:
  * Alterado `"turbulence_intensity"` de `0.6` para `0.15`. Isso impede que os puffs individuais se dispersem de forma muito caótica e irregular, mantendo a pluma como um "rasto" limpo.
* **Amplitude de Oscilação (Meandro OU) (config.json linhas 28-31 / main.py linha 1610)**:
  * Configurado `"ou_meander_config": {"intensity": 0.4, "timescale": 0.15}` para dar grande amplitude de movimento vertical de oscilação do vento (fazendo a pluma varrer o mapa).

### Ficheiro: [main.py](file:///C:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/main.py)
* **Espessura da Pluma (Linha 43)**:
  * No método Numba compilado JIT `update_puffs_numba()`, alterámos a fórmula de crescimento do raio de cada puff individual:
    * *Antes*: `puffs_array[i, 2] = 0.1 * np.sqrt(puffs_array[i, 4])`
    * *Agora*: `puffs_array[i, 2] = 0.2 * np.sqrt(puffs_array[i, 4])`
  * Multiplicar por `0.2` (em vez de `0.1`) faz com que cada partícula se expanda ao dobro do diâmetro ao longo do tempo, gerando uma pluma visual e fisicamente duas vezes mais grossa e muito mais fácil de cruzar e detetar pelo agente.

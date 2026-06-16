# Estimador de População para WSN — Documentação e Roadmap

Este documento descreve o estimador atual (Método 1) e esboça dois novos
métodos derivados dos notebooks P3, além de uma estratégia de integração
dos três.

---

## 1. Contexto geral

O problema de estimação de população (P2) pergunta:

> **Dado um problema de posicionamento de relays em uma WSN, qual é o
> tamanho mínimo de população `n` que um algoritmo evolutivo precisa para
> encontrar uma solução boa com probabilidade de pelo menos `1 - alpha`?**

Todos os três métodos abaixo estimam `n` pelo lado pessimista — preferem
superestimar do que subestimar, garantindo que a busca não seja abortada
cedo demais.

---

## 2. Método 1 — Estimativa por Gambler-Ruin (atual)

### 2.1 Ideia central

O estimador é derivado da aproximação do *gambler's ruin*: um algoritmo
evolutivo com população `n` converge para a solução ótima do bloco `i` com
probabilidade `≥ 1 − alpha` se `n ≥ n_i_hat`. O parâmetro decisivo é **o
quão difícil é o bloco** — medido pela diferença entre a qualidade da
configuração ótima e a de um competidor deceptivo.

### 2.2 Estrutura de blocos

O espaço de candidatos `Q` (de tamanho `J`) é partido em `B` blocos
`Q_1, …, Q_B` por um método configurável (`grid`, `kmeans`,
`radial_to_sink`). Para cada bloco `i`:

| símbolo      | significado                                                                        |
|--------------|------------------------------------------------------------------------------------|
| `k_i`        | tamanho do bloco `\|Q_i\|`                                                         |
| `H_i^*`      | configuração "ótima heurística" do bloco (método `structural_greedy`, etc.)        |
| `H_i^L`      | configuração "deceptiva" competidora (`deceptive_low_cost`, etc.)                  |
| `x_{-i}`     | complemento — bits fora do bloco `i`, amostrados aleatoriamente                   |
| `F(x)`       | objetivo escalarizado (connectivity, relay_count, hops, …)                         |
| `Delta_i^r`  | `F(H_i^*, x_{-i}^r) - F(H_i^L, x_{-i}^r)` para o `r`-ésimo complemento          |

### 2.3 Fórmulas do estimador

Após `R` avaliações de complementos aleatórios calcula-se:

```
d_i_hat     = média(Delta_i^r)        # gap de qualidade
sigma_BB_i  = desvio-padrão(Delta_i^r)  # variabilidade
```

Dois estimadores fechados derivam do mesmo framework teórico:

**Inicialização uniforme** (bits sorteados com probabilidade 1/2):

```
n_i_hat = -ln(alpha) * 2^(k_i - 1) * sigma_BB_i * sqrt(2*m) / d_i_hat
```

**Inicialização Bernoulli** (bits `iid Bernoulli(rho)`):

```
n_i_hat = (-ln(alpha) / (2 * pi_i(H_i^*))) * sigma_BB_i * sqrt(2*m) / d_i_hat

  com  pi_i(H_i^*) = rho^s_i * (1-rho)^(k_i - s_i)
```

onde `m` é o número de posições amostradas nas trajetórias e `s_i` é o
número de bits ativos em `H_i^*`.

### 2.4 Agregação global

A estimativa final é o pior caso entre os blocos válidos:

```
n_hat = max_i  ceil(n_i_hat)         (blocos com status "ok" ou "degenerate_variance")
```

Blocos com `d_i <= 0` (H_star não é melhor que H_local) ou com menos de 2
amostras são marcados como inválidos e excluídos.

### 2.5 Como usar (CLI)

```bash
# Modo surrogate (sem Cooja)
python -m p2_population_estimator run_surrogate.json

# Modo Cooja (simulação real via SSH)
python -m p2_population_estimator run_cooja.json
```

Campos obrigatórios no JSON de configuração:

```json
{
  "instance_path":    "examples/ind2.json",
  "output_dir":       "results/minha_rodada",
  "mode":             "surrogate",
  "partition_method": "grid",
  "num_blocks":       8,
  "num_complements":  30,
  "alpha":            0.05,
  "rho":              0.5,
  "seeds":            [42, 43, 44]
}
```

Resultados gravados em `output_dir/`:

```
population_estimate_result.json   # resumo global + per-bloco
block_results.csv                 # tabela plana para análise
```

---

## 3. Método 2 — Estimativa por Cobertura de Adjacência Temporal (esboço)

### 3.1 Ideia central

O módulo `adjacency_builder.py` produz, para cada instância, a matriz
**acumulada de co-ocorrência**:

```
A_total[u, v] = #{t : dist(u, v, t) <= R}
```

A partir dessa matriz define-se o **score de cobertura móvel** de cada
candidato fixo `u`:

```
cov(u) = sum_{m in mobiles} A_total[u, m]     ∈ [0, T*M]
```

`cov(u)` mede por quantos "móvel-instantes" o candidato `u` estava ao
alcance de algum sensor móvel. Um candidato com `cov(u) = 0` nunca cobre
tráfego móvel e pode ser descartado; candidatos com `cov(u)` alto são
estruturalmente necessários.

### 3.2 Hipótese de estimação

> O número mínimo de relays ativos para garantir cobertura ao longo de
> toda a trajetória é limitado inferiormente pelo número de candidatos
> "insubstituíveis" — aqueles cujo `cov(u)` supera um limiar `theta`.

Formalmente, dado um limiar `theta`:

```
C_theta = #{u in Q : cov(u) >= theta}
```

A estimativa de população é:

```
n_hat_cov(theta) = f(C_theta, k_bar, alpha)
```

onde `k_bar` é o tamanho médio dos blocos (proxy do espaço de busca por
bloco) e `f` é uma função monotônica a ser calibrada empiricamente.

Uma forma simples de calibração inicial:

```
n_hat_cov = C_theta * (1 / alpha)^(1 / k_bar)
```

A intuição é: com `C_theta` candidatos obrigatórios espalhados por blocos
de tamanho `k_bar`, a probabilidade de um individuo aleatório tê-los todos
corretos é `~(1/2)^C_theta`, logo precisamos de `n ≈ (1/alpha)^(1/k_bar)`
indivíduos por região crítica.

### 3.3 Como usar (pipeline proposto)

```python
import adjacency_builder as ab

instance = ab.load_instance("examples/ind2.json")
result   = ab.build_from_instance(instance)

layout   = result["layout"]
acc      = result["accumulated"]          # (K, K) int64
m_idx    = layout.mobile_indices
c_idx    = layout.candidate_indices

import numpy as np
cov = acc[np.ix_(c_idx, m_idx)].sum(axis=1)  # (N,) cobertura de cada candidato

T, M = result["tensor"].shape[0], layout.M
theta = 0.1 * T * M                          # 10% do máximo como exemplo
C_theta = int((cov >= theta).sum())
print(f"Candidatos com cobertura >= {theta:.0f}: {C_theta}")
```

### 3.4 Parâmetros a calibrar

| parâmetro | papel                                          | sugestão inicial           |
|-----------|------------------------------------------------|----------------------------|
| `theta`   | limiar de cobertura mínima                     | 5–20% de `T*M`             |
| `f`       | mapeamento `C_theta -> n_hat`                  | linear escalado por `alpha` |

---

## 4. Método 3 — Estimativa por Uso em Menores Caminhos (esboço)

### 4.1 Ideia central

O módulo `path_builder.py` computa, para cada instante `t`, os menores
caminhos BFS do sink a cada sensor móvel e acumula quantas vezes cada nó
e aresta foram utilizados:

```
node_accumulated[u] = sum_t #{m : u pertence ao menor caminho m->sink no instante t}
```

Para um relay candidato, `node_accumulated[u]` captura sua **importância
operacional de roteamento** — não apenas se ele está no alcance, mas se ele
é *de fato usado* para rotear tráfego de volta ao sink.

### 4.2 Hipótese de estimação

> O número de relays cuja remoção quebraria os menores caminhos constitui
> um limite inferior rígido na população necessária.

Define-se o **score de roteamento**:

```
route(u) = node_accumulated[u] / (T * M)     ∈ [0, 1]
```

e o conjunto de relays críticos:

```
R_phi = {u in Q : route(u) >= phi}
```

A estimativa de população é:

```
n_hat_route(phi) = g(|R_phi|, k_bar, alpha)
```

Uma forma natural é usar a mesma estrutura do Método 1, mas substituindo a
análise de blocos pela criticidade de roteamento. Para cada bloco `i`
definem-se:

- `rho_i^route` = fração de candidatos críticos no bloco: `|R_phi ∩ Q_i| / k_i`
- `d_i^route` = distância esperada entre configurações com e sem os relays críticos

Estimativa por bloco (análoga à fórmula uniforme do Método 1):

```
n_i_hat_route = -ln(alpha) * (1 / rho_i^route) * sigma_route_i * sqrt(2*m) / d_i^route
```

### 4.3 Como usar (pipeline proposto)

```python
import adjacency_builder as ab
import path_builder as pb
import numpy as np

instance = ab.load_instance("examples/ind2.json")
result   = pb.build_from_instance(instance)

layout       = result["layout"]
node_acc     = result["node_accumulated"]       # (K,) int64
T, M         = result["node_per_t"].shape[0], layout.M
c_idx        = np.asarray(layout.candidate_indices)

route_score  = node_acc[c_idx] / (T * M)       # (N,) ∈ [0, 1]
phi          = 0.05                             # 5% do máximo
R_phi        = int((route_score >= phi).sum())
print(f"Candidatos críticos de roteamento (phi={phi}): {R_phi}")

# Exportar para usar em análise comparativa
pb.export_results("results/p3_route_demo", result, pb.summarise(result, instance), force=True)
```

### 4.4 Parâmetros a calibrar

| parâmetro | papel                                        | sugestão inicial            |
|-----------|----------------------------------------------|-----------------------------|
| `phi`     | limiar de score de roteamento                | 1–10% de `T*M`              |
| `g`       | mapeamento `\|R_phi\| -> n_hat`              | análogo ao Método 1         |

---

## 5. Integração dos três métodos (segundo passo)

### 5.1 Arquitetura proposta

Os três métodos capturam aspectos complementares da dificuldade do problema:

| método  | perspectiva         | dado primário              | tipo de bound  |
|---------|---------------------|----------------------------|----------------|
| M1      | dificuldade de busca | `d_i`, `sigma_BB_i`       | probabilístico |
| M2      | cobertura geométrica | `A_total[u, m]`           | estrutural     |
| M3      | fluxo de roteamento  | `node_accumulated[u]`     | operacional    |

A integração pode seguir três estratégias, da mais conservadora à mais sofisticada:

#### Estratégia A — Máximo conservador

```
n_hat_integrated = max(n_hat_M1, n_hat_M2, n_hat_M3)
```

Usa o pior caso entre os três. Garante que nenhuma perspectiva é ignorada.
Adequado como baseline antes de calibração.

#### Estratégia B — Média ponderada calibrada

```
n_hat_integrated = w1 * n_hat_M1 + w2 * n_hat_M2 + w3 * n_hat_M3
    com  w1 + w2 + w3 = 1
```

Pesos `w1, w2, w3` calibrados por regressão sobre experimentos com `n`
verdadeiro conhecido. O M1 tende a dominar em instâncias com blocos
deceptivos fortes; o M3 domina em instâncias com gargalos topológicos.

#### Estratégia C — Meta-estimador por bloco

Para cada bloco `i`, combinar as três perspectivas ao nível do bloco:

```
n_i_hat_meta = h(n_i_hat_M1, cov_i, route_i, alpha, k_i)
```

onde:
- `cov_i = mean_{u in Q_i} cov(u) / (T*M)` — cobertura média do bloco
- `route_i = mean_{u in Q_i} route(u)` — importância de roteamento média do bloco
- `h` é uma função combinada a ser aprendida ou derivada teoricamente

O meta-estimador por bloco é o mais expressivo: blocos com alta cobertura
E alta importância de roteamento recebem estimativas maiores, pois errar
nesses blocos é mais custoso.

### 5.2 Diagrama de fluxo

```
Instance JSON
     |
     ├─── [M1] Experiment.run()
     │         │
     │         └── BlockComparisonResult per bloco
     │                  d_i, sigma_BB_i, n_i_uniform, n_i_bernoulli
     │
     ├─── [M2] adjacency_builder.build_from_instance()
     │         │
     │         └── A_total (K×K)
     │                  cov(u) per candidato, C_theta per bloco
     │
     └─── [M3] path_builder.build_from_instance()
               │
               └── node_accumulated (K,)
                        route(u) per candidato, R_phi per bloco
                                    │
                                    ▼
                           Integrator (Estratégia A/B/C)
                                    │
                                    ▼
                         n_hat_integrated  +  diagnóstico por bloco
```

### 5.3 Implementação sugerida (estrutura de módulos)

```
pop-estimator/
  p2_population_estimator/
    estimator.py          ← Método 1 (existente)
    coverage_estimator.py ← Método 2 (novo)  usa adjacency_builder
    routing_estimator.py  ← Método 3 (novo)  usa path_builder
    integrator.py         ← combina M1 + M2 + M3
    models.py             ← adicionar CoverageEstimateResult, RoutingEstimateResult
```

#### Interface mínima para `coverage_estimator.py`

```python
def estimate_from_coverage(
    accumulated: np.ndarray,
    layout: NodeLayout,
    blocks: list[CandidateBlock],
    *,
    alpha: float,
    theta_fraction: float = 0.10,  # theta = theta_fraction * T * M
    T: int,
) -> list[CoverageBlockResult]:
    ...

def aggregate_coverage(results: list[CoverageBlockResult]) -> dict[str, object]:
    ...
```

#### Interface mínima para `routing_estimator.py`

```python
def estimate_from_routing(
    node_accumulated: np.ndarray,
    layout: NodeLayout,
    blocks: list[CandidateBlock],
    *,
    alpha: float,
    phi: float = 0.05,
    T: int,
) -> list[RoutingBlockResult]:
    ...

def aggregate_routing(results: list[RoutingBlockResult]) -> dict[str, object]:
    ...
```

#### Interface mínima para `integrator.py`

```python
def integrate(
    m1: dict,   # saída de aggregate_global()
    m2: dict,   # saída de aggregate_coverage()
    m3: dict,   # saída de aggregate_routing()
    strategy: Literal["max", "weighted", "per_block"] = "max",
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, object]:
    ...
```

### 5.4 Próximos passos

1. **Implementar `coverage_estimator.py`** usando `adjacency_builder` como dependência externa ao pacote `p2_population_estimator`.
2. **Implementar `routing_estimator.py`** usando `path_builder`.
3. **Calibrar os limiares `theta` e `phi`** com as instâncias existentes em `pop-estimator/examples/` usando o modo surrogate.
4. **Validar com Cooja** comparando `n_hat_M1`, `n_hat_M2`, `n_hat_M3` e `n_hat_integrated` contra populações observadas experimentalmente.
5. **Implementar `integrator.py`** começando pela Estratégia A (máximo) e avançando para B ou C após calibração.

---

## 6. Referências internas

| arquivo                            | papel                                         |
|------------------------------------|-----------------------------------------------|
| `p2_population_estimator/estimator.py`    | fórmulas gambler-ruin (M1)             |
| `p2_population_estimator/blocks.py`       | heurísticas H_star e H_local           |
| `p2_population_estimator/statistics.py`   | d_hat, sigma_BB_hat, delta_samples     |
| `p2_population_estimator/experiment.py`   | orquestração do loop de avaliação      |
| `adjacency_builder.py`             | A(t), A_total, scores de cobertura (M2)       |
| `path_builder.py`                  | menores caminhos BFS, node_accumulated (M3)   |
| `p3-building-blocks.ipynb`         | exploração visual de A_total                  |
| `p3-path-blocks.ipynb`             | exploração visual de node_accumulated         |

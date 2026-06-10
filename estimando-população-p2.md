# Estimando População problema P2

## Definição problema P2

**Dados**
1. uma região $\Omega\subset\mathbb{R}^2$ (compacta e conexa)
2. um ponto fixo $s\in\Omega$ onde será instalado o sink
3. um conjunto finito $Q=\{\xi_1,\dots,\xi_J\}\subset\Omega$ de posições candidata para os sensores/retransmissores
4. um conjunto de $M$ de sensores móveis
$$
\Gamma=\{\gamma_m:\mathbb{R}_+\rightarrow\mathbb{R}^2\}_{m-1}^M
$$
5. um raio de alcance para comunicação entre dois dispositivos $R>0$

**O problema consiste em**

Determinar o menor subconjunto $P\subset Q$ que para todo instânte $t\in\mathbb{R}_+$, cada sensor móvel $\gamma_m$ consiga se comunicar com o sink $s$ por meio de uma sequencia de sensores conectados por estarem dentro do raio de comunicação uns dos outros.

Além disso, desejamos encontrar a fronteira de pareto para três métricas de rede.

## Modelo do Cromossomo

Um cromossomo para este problema é uma tupla binária $B\in\{0,1\}^{|Q|}$ onde cada coordenada indica se o candidato correspondente dever ser instalado "1" ou não "0".

## Estimativa da população

Partiremos da fórmula

$$
n \approx -\ln(\alpha)2^{k-1}\frac{\sigma_{BB}\sqrt{2m}}{d}.
$$

Ela estima o tamanho populacional $n$ necessário para reduzir a probabilidade $\alpha$ de perder uma instância ótima de bloco durante a seleção. Essa fórmula vem da aproximação por ruína do jogador, em que $p$ é a probabilidade de a instância correta vencer, $q=1-p$, e a população inicial contém aproximadamente $a=n/2^k$ cópias esperadas de uma determinada instância de bloco de tamanho $k$.  A dedução final identifica $\alpha=1-P_n$ como probabilidade de fracasso e chega à expressão acima usando $x=d/(\sigma_{BB}\sqrt{2m})$. 

O problema P2 envolve selecionar candidatos para cobrir trajetórias móveis e manter conectividade até o sink. Então, diferentemente de uma função Trap-(k), os "blocos" não são naturalmente posições consecutivas no cromossomo. Eles precisam ser **construídos semanticamente**.

A pergunta principal é:

> O que seria um bloco construtivo no P2?


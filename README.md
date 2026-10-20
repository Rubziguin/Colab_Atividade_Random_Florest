[README.md](https://github.com/user-attachments/files/33730421/README.md)
# IA ML — Random Forest com Scikit-Learn

**PUC-SP · Ciência da Computação · Inteligência Artificial**
Atividade do LAB da aula *AI ML Random Forest*


## Dupla

- Rubens Rodrigues Maranesi
- Pedro Gabriel Takenobu Serafim

## Sobre

Exploramos o modelo **Random Forest** do scikit-learn em dois conjuntos de dados:

| Dataset | Fonte | Tarefas |
|---|---|---|
| **Titanic** | [Kaggle](https://www.kaggle.com/c/titanic/data) | Classificação: sobreviveu ou não |
| **cars** | [Vega Datasets](https://github.com/vega/vega-datasets) | Classificação: país de origem (USA, Europe, Japan) · Regressão: consumo (milhas por galão) |

O dataset `cars` tem 406 carros fabricados entre 1970 e 1982, com cilindros, cilindrada, potência, peso, aceleração, ano, consumo e origem.

## Itens do LAB

| # | Item | Seção do notebook |
|---|---|---|
| 1 | Você sobreviveria ao Titanic? | A.5 |
| 2 | Estatística descritiva e interpolação para preencher `Age` | A.2 |
| 3 | Limpar melhor os dados do Titanic | A.3 |
| 4 | Usar `test.csv` e verificar a acurácia | A.4 |
| 5 | Random Forest com outro dataset do Vega (`cars`) | B.1 – B.2 |
| 6 | Explicar IA, ML e Random Forest | C.1 |
| 7 | Experimentar regressão | B.3 |
| 8 | Visualizações | todo o notebook, com destaque para B.4 |
| 9 | Analisar o gráfico do Random Forest | B.5 |
| 10 | Prós e contras: Decision Tree × Random Forest | C.2 |

## Principais resultados

### Titanic

| Modelo | Acurácia no treino | Acurácia em dados novos (CV 5 folds) |
|---|---|---|
| Modelo da aula (Pclass, Sex, Age) | 0,880 | 0,792 |
| Dados limpos + novas features, RF padrão | 0,984 | 0,807 |
| Dados limpos + novas features, RF ajustado | 0,899 | **0,838** |

- **Preenchimento de `Age`:** média, mediana, moda e média por sexo ficaram praticamente empatadas (0,79–0,80). Já sortear idades ou interpolar piorou o resultado (≈0,77), porque coloca ruído nos dados.
- **Limpeza:** extraímos o título do nome (Mr, Mrs, Miss, Master), criamos tamanho da família, presença de cabine, deck e porto de embarque, e preenchemos a idade pela mediana de cada título. Também corrigimos dois problemas do tratamento da aula: o `fillna(29)` preenchia todas as colunas, e a idade dos bebês era multiplicada por 10.
- **Ajuste de hiperparâmetros:** com `GridSearchCV`, o melhor resultado foi `min_samples_leaf=4` e `max_features=0.5`. Isso diminuiu o overfitting (o modelo deixou de decorar o treino).
- **Teste:** o `test.csv` do Kaggle não tem a coluna `Survived`. Por isso, separamos 20% do treino como teste e o modelo acertou **81,6%**. Também geramos o `submission.csv` para enviar ao Kaggle.
- **Nós sobreviveríamos?** Rubens (1ª classe, 25 anos): **30%** de chance. Pedro (3ª classe, 21 anos): **6%**. Nenhum dos dois sobreviveria. O modelo mostra claramente a regra "mulheres e crianças primeiro": uma mulher de 22 anos na 1ª classe teria 97% de chance.

### cars: de onde vem o carro?

| Modelo | Acurácia no teste | Acurácia CV 10 folds |
|---|---|---|
| Chutar sempre "USA" | 0,627 | — |
| Decision Tree | 0,794 | 0,842 ± 0,072 |
| Random Forest | **0,863** | **0,847 ± 0,047** |

Carros americanos são fáceis de identificar (grandes, pesados, com 8 cilindros). A confusão do modelo fica entre **Europa e Japão**, que fabricavam carros pequenos muito parecidos. A cilindrada (`Displacement`) foi a feature mais importante.

### cars: regressão do consumo

| Modelo | MAE (mpg) | RMSE (mpg) | R² teste | R² CV 10 folds |
|---|---|---|---|---|
| Regressão Linear | 2,33 | 2,96 | 0,846 | 0,815 |
| Decision Tree | 2,48 | 3,56 | 0,776 | 0,753 |
| Random Forest | **1,68** | **2,43** | **0,896** | **0,880** |

O erro do Random Forest cai rapidamente até cerca de 20 árvores e depois se estabiliza. Mais árvores não causam overfitting, apenas deixam o treino mais lento.

## Decision Tree × Random Forest

| | Decision Tree | Random Forest |
|---|---|---|
| Interpretabilidade | Fácil de desenhar e explicar | Centenas de árvores, mais difícil de interpretar |
| Precisão em dados novos | Menor | Maior |
| Overfitting | Alto sem limitar a profundidade | Bem menor, porque a média das árvores reduz a variância |
| Estabilidade | Muda muito com pequenas mudanças nos dados | Estável |
| Tempo de treino (cars) | ≈ 4 ms | ≈ 0,9 s |

## Como executar

**No Google Colab:** clique no botão "Abrir no Colab" no topo e use *Ambiente de execução → Executar tudo*.

**Localmente:**

```bash
pip install scikit-learn pandas numpy matplotlib seaborn missingno vega_datasets kagglehub jupyter
jupyter notebook PUCSP_IA_ML_RandomForest_Atividade.ipynb
```

Os dados do Titanic são baixados do Kaggle com o `kagglehub` (a chave fica em *Secrets* do Colab como `KAGGLE_API_TOKEN`). Sem a chave, o notebook usa automaticamente uma cópia pública dos mesmos arquivos `train.csv` e `test.csv`. O dataset `cars` já vem dentro do pacote `vega_datasets`.

## Tecnologias

Python · scikit-learn · pandas · NumPy · Matplotlib · seaborn · missingno · vega_datasets · kagglehub

## Referências

- [Documentação do scikit-learn](https://scikit-learn.org/stable/)
- [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Árvores de decisão no scikit-learn](https://scikit-learn.org/stable/modules/tree.html)
- [Kaggle — Titanic](https://www.kaggle.com/c/titanic)
- [Vega Datasets](https://github.com/vega/vega-datasets)
- James, Witten, Hastie, Tibshirani — [An Introduction to Statistical Learning](https://www.statlearning.com/)

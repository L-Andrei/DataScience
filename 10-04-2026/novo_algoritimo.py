import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

#Preparação dos Dados
df = pd.read_csv('itau.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Criar features antes de qualquer limpeza
df['SMA_5'] = df['Price'].rolling(5).mean()
df['Retorno_Hoje'] = df['Price'].pct_change()

# Capturamos o dia anterior do dropna 
features = ['Price', 'Open', 'High', 'Low', 'SMA_5', 'Retorno_Hoje']
# Pegamos a última linha
dados_ontem = df[features].iloc[[-1]]

# O alvo é a diferença para o dia seguinte (que ainda não existe para o último dia)
df['Diff'] = df['Price'].shift(-1) - df['Price']

# Agora limpamos para o treino (isso remove a linha de ontem do X_train/X_test)
df_treino = df.dropna()

X = df_treino[features]
y = df_treino['Diff']

#Divisão 90% Treino e 10% Teste
split_idx = int(len(X) * 0.9)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

#Treinamento
model = LinearRegression()
model.fit(X_train, y_train)

#Verificação de Eficiência
diff_preds = model.predict(X_test)
precos_reais = X_test['Price'] + y_test
precos_previstos = X_test['Price'] + diff_preds

mae_final = root_mean_squared_error(precos_reais, precos_previstos)
baseline_mae = root_mean_squared_error(precos_reais, X_test['Price'])

print("--- RELATÓRIO: REGRESSÃO LINEAR (10/04) ---")
print(f"MAE Final: R$ {mae_final:.2f}")
print(f"MAE Baseline: R$ {baseline_mae:.2f}")
print(f"R² Score: {r2_score(precos_reais, precos_previstos):.4f}")


variacao_prevista = model.predict(dados_ontem)[0]
preco_ontem = dados_ontem['Price'].values[0]

print(f"Variação esperada para hoje: R$ {variacao_prevista:+.2f}")
print(f"--- PREVISÃO PARA 10/04: R$ {preco_ontem + variacao_prevista:.2f} ---")

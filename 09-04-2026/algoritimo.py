import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Preparação dos Dados
df = pd.read_csv('EMBRAER_final.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# --- MUDANÇA AQUI: Isolar os dados de ontem (08/04) ANTES do dropna ---
features = ['Price', 'Open', 'High', 'Low', 'Gap_Price_Open', 'Gap_High_Low']
dados_para_prever_hoje = df[features].iloc[[-1]] 

# Criar o alvo (Target) para o treino
df['Target'] = df['Price'].shift(-1)

# Agora sim, limpamos o dataset de treino (remove a última linha que não tem target)
df_treino = df.dropna()

X = df_treino[features]
y = df_treino['Target']

# 2. Divisão 90% Treino e 10% Teste
split_idx = int(len(X) * 0.9)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 3. Treinamento
model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train)

# 4. Verificação
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Erro Médio Absoluto (MAE): R$ {mae:.2f}")

# 5. Previsão para HOJE (09/04)
# Usamos os dados que isolamos lá no início
previsao_hoje = model.predict(dados_para_prever_hoje)

print(f"\nDados de entrada (ontem - 08/04):")
print(dados_para_prever_hoje)
print(f"\nPrevisão estimada para o fechamento de hoje (09/04): R$ {previsao_hoje[0]:.2f}")
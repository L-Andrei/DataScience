import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Preparação dos Dados
df = pd.read_csv('EMBRAER_final.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date') # Garantir ordem cronológica

# Criar o alvo (Target): Preço do dia seguinte
df['Target'] = df['Price'].shift(-1)
df = df.dropna() # Remove a última linha que não tem o alvo

# Selecionar Features (X) e Alvo (y)
features = ['Price', 'Open', 'High', 'Low', 'Gap_Price_Open', 'Gap_High_Low']
X = df[features]
y = df['Target']

# 2. Divisão 90% Treino e 10% Teste (Sem Shuffle!)
split_idx = int(len(df) * 0.9)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 3. Treinamento do Modelo
model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train)

# 4. Verificação de Eficiência
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"Erro Médio Absoluto (MAE): R$ {mae:.2f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): R$ {rmse:.2f}")

# 5. Previsão para o dia 08/04
# Usamos os dados do último dia disponível (07/04) para prever o dia 08/04
ultimo_dia_dados = X.iloc[[-1]]
previsao_amanha = model.predict(ultimo_dia_dados)

print(f"\nPrevisão estimada para o fechamento de 08/04: R$ {previsao_amanha[0]:.2f}")
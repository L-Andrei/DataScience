import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

#Preparação dos Dados
df = pd.read_csv('itau.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# O alvo continua sendo a diferença para evitar o problema de escala
df['Diff'] = df['Price'].shift(-1) - df['Price']

df['SMA_5'] = df['Price'].rolling(5).mean()
df['Retorno_Hoje'] = df['Price'].pct_change()
df = df.dropna()

features = ['Price', 'Open', 'High', 'Low', 'SMA_5', 'Retorno_Hoje']
X = df[features]
y = df['Diff']

#Divisão 90/10
split_idx = int(len(df) * 0.9)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

#Treinamento 
model = LinearRegression()
model.fit(X_train, y_train)

#Verificação
diff_preds = model.predict(X_test)
precos_reais = X_test['Price'] + y_test
precos_previstos = X_test['Price'] + diff_preds

mae_final = root_mean_squared_error(precos_reais, precos_previstos)
baseline_mae = root_mean_squared_error(precos_reais, X_test['Price'])

print("--- RELATÓRIO: REGRESSÃO LINEAR ---")
print(f"MAE Final: R$ {mae_final:.2f}")
print(f"MAE Baseline: R$ {baseline_mae:.2f}")
print(f"R² Score: {r2_score(precos_reais, precos_previstos):.4f}")

#Previsão para Amanhã
ultimo_dia = X.iloc[[-1]]
variacao_prevista = model.predict(ultimo_dia)[0]
preco_atual = ultimo_dia['Price'].values[0]

print(f"Variação prevista: R$ {variacao_prevista:+.2f}")
print(f"Tendência para amanhã: {'ALTA' if variacao_prevista > 0 else 'BAIXA'}")
print(f"Previsão estimada: R$ {preco_atual + variacao_prevista:.2f}")

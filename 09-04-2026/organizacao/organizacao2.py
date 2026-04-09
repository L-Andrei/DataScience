import pandas as pd

# 1. Carregar o dataset processado
df = pd.read_csv('EMBRAER_processado.csv')

# 2. Arredondar as colunas de Gap para 2 casas decimais
df['Gap_Price_Open'] = df['Gap_Price_Open'].round(2)
df['Gap_High_Low'] = df['Gap_High_Low'].round(2)

# 3. Salvar o arquivo final simplificado
df.to_csv('EMBRAER_final.csv', index=False)

print("Colunas simplificadas com sucesso!")
print(df[['Date', 'Gap_Price_Open', 'Gap_High_Low']].head())
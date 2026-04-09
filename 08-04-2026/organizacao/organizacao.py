import pandas as pd

# 1. Carregar o dataset original
df = pd.read_csv('embraer.csv')

# 2. Remover a coluna 'Change %' (a menos impactante)
if 'Change %' in df.columns:
    df = df.drop(columns=['Change %'])

# 3. Criar as variáveis de Gap (Diferença entre preços)
# Gap de Fechamento: Indica a força do movimento intradiário
df['Gap_Price_Open'] = df['Price'] - df['Open']

# Gap High-Low: Indica a volatilidade (amplitude) do dia
df['Gap_High_Low'] = df['High'] - df['Low']

# 4. Salvar o novo arquivo processado
df.to_csv('EMBRAER_processado.csv', index=False)

# Exibir o resultado final para conferência
print("Processamento concluído. Novas colunas adicionadas:")
print(df[['Date', 'Price', 'Open', 'Gap_Price_Open', 'Gap_High_Low']].head())
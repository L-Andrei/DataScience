import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar e preparar os dados
df = pd.read_csv('itau.csv')
if 'Unnamed: 7' in df.columns:
    df = df.drop(columns=['Unnamed: 7'])

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Formatar as datas para o eixo X
labels = df['Date'].dt.strftime('%d/%m/%y').tolist()
x = np.arange(len(labels))
width = 0.25

# Criar o gráfico de barras
fig, ax = plt.subplots(figsize=(12, 7))

# Plotagem das três barras para cada dia
rects1 = ax.bar(x - width, df['Open'], width, label='Abertura (Open)', color='#3498db', alpha=0.8)
rects2 = ax.bar(x, df['Price'], width, label='Fechamento (Price)', color='#2ecc71', alpha=0.8)
rects3 = ax.bar(x + width, df['Resultado'], width, label='Resultado (Predição)', color='#e67e22', alpha=0.8)

# Configurações do gráfico
ax.set_ylabel('Preço (R$)')
ax.set_title('Comparação Diária: Abertura, Fechamento e Predição')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Função para adicionar os valores decimais sobre as barras
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Ajuste do eixo Y para melhor visualização das diferenças
ymin = min(df['Open'].min(), df['Price'].min(), df['Resultado'].min()) - 1
ymax = max(df['Open'].max(), df['Price'].max(), df['Resultado'].max()) + 2
ax.set_ylim(ymin, ymax)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('comparativo_barras.png')

# 3. Cálculos de métricas (Direção e Erro Médio)
actual_move = df['Price'] - df['Open']
predicted_move = df['Resultado'] - df['Open']

def check_direction(actual, predicted):
    # Verifica se ambos subiram ou ambos caíram em relação à abertura
    if (actual > 0 and predicted > 0) or (actual < 0 and predicted < 0) or (actual == 0 and predicted == 0):
        return True
    return False

hits = sum([check_direction(a, p) for a, p in zip(actual_move, predicted_move)])
total = len(df)
accuracy = (hits / total) * 100
avg_diff = np.sqrt(((np.abs(df['Price'] - df['Resultado']))**2).mean())

print(f"Direções acertadas: {hits} de {total} ({accuracy:.2f}%)")
print(f"Valor em reais médio de diferença: R$ {avg_diff:.2f}")

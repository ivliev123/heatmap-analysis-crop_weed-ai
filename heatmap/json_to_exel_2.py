import json
import pandas as pd

# Загрузка JSON-файла
with open("metrics_main.json", "r") as f:
    data = json.load(f)

# Преобразование в DataFrame
df = pd.DataFrame.from_dict(data, orient="index")

# Сброс индекса и переименование колонки с названиями моделей
df.reset_index(inplace=True)
df.rename(columns={"index": "Model"}, inplace=True)

# Сохранение в Excel
output_file = "metrics_main_table.xlsx"
df.to_excel(output_file, index=False)

print(f"Таблица успешно сохранена в файл: {output_file}")

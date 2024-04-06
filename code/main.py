import csv
import json


def csv_to_jsonl(csv_file, jsonl_file):
    with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过标题行
        with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
            for row in reader:
                file_name = row[0]
                text = row[1]
                data = {
                    "file_name": file_name,
                    "text": text
                }
                jsonlfile.write(json.dumps(data) + '\n')


# 读取CSV文件并转换为JSONL文件
csv_file = './data.csv'
jsonl_file = 'data.jsonl'

csv_to_jsonl(csv_file, jsonl_file)

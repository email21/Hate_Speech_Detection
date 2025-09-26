import csv
import json

def convert_csv_to_json(input_file, output_file):
    """
    CSV 파일을 읽어 각 행을 JSON 객체로 변환하여 새로운 텍스트 파일에 저장합니다.
    
    Args:
        input_file (str): 변환할 CSV 파일 경로 (예: 'result.csv')
        output_file (str): JSON 객체를 저장할 출력 파일 경로 (예: 'output.jsonl')
    """
    try:
        with open(input_file, mode='r', encoding='utf-8') as infile, \
             open(output_file, mode='w', encoding='utf-8') as outfile:
            
            # CSV 파일을 사전(Dictionary) 형태로 읽어옵니다.
            reader = csv.DictReader(infile)
            
            # 각 행을 순회하며 JSON으로 변환합니다.
            for row in reader:
                # 'output' 필드 값을 문자열에서 정수로 변환합니다.
                row['output'] = int(row['output'])
                
                # 딕셔너리를 JSON 문자열로 변환하고 줄바꿈 문자를 추가합니다.
                # 이는 JSON Lines(JSONL) 형식에 해당합니다.
                json_string = json.dumps(row, ensure_ascii=False)
                outfile.write(json_string + '\n')
                
        print(f"'{input_file}' 파일이 '{output_file}' 파일로 성공적으로 변환되었습니다.")
        
    except FileNotFoundError:
        print(f"오류: 파일 '{input_file}'을(를) 찾을 수 없습니다. 파일 경로를 확인해 주세요.")
    except Exception as e:
        print(f"변환 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    # 변환할 파일 경로와 저장할 파일 경로를 지정합니다.
    input_csv_file = 'result.csv'
    output_jsonl_file = 'output.jsonl'
    
    convert_csv_to_json(input_csv_file, output_jsonl_file)
import json
import csv
import os

# 변환할 대상 파일들을 리스트로 정의
input_files = [
    '../NIKL_AU_2023_COMPETITION_v1.0/nikluge-au-2022-test.jsonl',
    '../NIKL_AU_2023_COMPETITION_v1.0/nikluge-au-2022-train.jsonl',
    '../NIKL_AU_2023_COMPETITION_v1.0/nikluge-au-2022-dev.jsonl'
]
# 출력 경로 지정. data.py 파일에서 정의된 파일명으로 변경.
output_files = [
    '../NIKL_AU_2023_COMPETITION_v1.0/test.csv',
    '../NIKL_AU_2023_COMPETITION_v1.0/train.csv',
    '../NIKL_AU_2023_COMPETITION_v1.0/dev.csv'
]

# 한 번에 3개 파일 처리하는 반복문
for input_file, output_file in zip(input_files, output_files):
    # 입력 파일 존재 확인
    if not os.path.exists(input_file):
        print(f"경고: '{input_file}' 파일을 찾을 수 없습니다. 건너뜁니다.")
        continue
        
    data = []
    fieldnames = None

    try:
        # JSONL 파일 읽기
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    
                    # 첫 번째 레코드에서 필드명 결정
                    if fieldnames is None:
                        available_fields = list(record.keys())
                        print(f"'{input_file}'의 필드들: {available_fields}")
                        
                        # test 파일인지 확인 (output 필드 없음)
                        if 'test' in input_file and 'output' not in available_fields:
                            fieldnames = ['id', 'input']  # test 파일은 id, input만
                            print("test 파일로 인식: id, input 필드만 사용")
                        else:
                            fieldnames = ['id', 'input', 'output']  # train/dev 파일
                            print("train/dev 파일로 인식: id, input, output 필드 사용")
                    
                    # 필수 필드 확인
                    if not all(key in record for key in fieldnames):
                        print(f"경고: {input_file}의 {line_num}번째 줄에 필요한 필드가 누락되었습니다.")
                        print(f"필요한 필드: {fieldnames}, 실제 필드: {list(record.keys())}")
                        continue
                    
                    # 필요한 필드만 선택해서 저장
                    filtered_record = {field: record[field] for field in fieldnames}
                    data.append(filtered_record)
                    
                except json.JSONDecodeError as e:
                    print(f"경고: {input_file}의 {line_num}번째 줄에서 JSON 파싱 오류: {e}")
                    continue

        if not data:
            print(f"경고: '{input_file}'에서 유효한 데이터를 찾을 수 없습니다.")
            continue

        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # CSV 파일로 쓰기(저장)
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        print(f"'{input_file}' 파일이 '{output_file}' 파일로 변환되었습니다. ({len(data)}개 레코드)")
        
    except Exception as e:
        print(f"오류: '{input_file}' 처리 중 오류 발생: {e}")
        continue

print("변환 작업이 완료되었습니다.")
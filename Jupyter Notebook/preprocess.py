# preprocess.py  'Jupyter Notebook'폴더 내에서 실행.

import pandas as pd
import os
import re
from tqdm import tqdm

# pandas의 progress_apply를 사용하기 위한 설정
tqdm.pandas()


def preprocess(text):
    """정규식을 사용한 텍스트 전처리 함수."""
    try:
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # 빈 문자열 체크
        if not text.strip():
            return ""

        # 1단계: 한글, 영어, 숫자, 그리고 지정된 특수문자들(.,!?&)을 제외한 모든 문자를 공백으로 변경
        text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9.,!?&\s]", " ", text)

        # 2단계: 최종적으로 여러 개의 공백을 하나로 정리
        text = re.sub(r"\s+", " ", text).strip()

        return text
    
    except Exception:
        return ""


def validate_dataframe(df, required_columns):
    """데이터프레임의 유효성을 검사합니다."""
    if df.empty:
        raise ValueError("데이터프레임이 비어있습니다.")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
    
    return True


def main():
    """CSV 파일을 전처리하고 기존 파일을 덮어씁니다."""
    # 현재 스크립트의 디렉토리를 기준으로 상대 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "..", "NIKL_AU_2023_COMPETITION_v1.0")
    dataset_dir = os.path.abspath(dataset_dir)
    
    filenames = ["train.csv"]
    
    print(f"데이터셋 디렉토리: {dataset_dir}")
    
    if not os.path.exists(dataset_dir):
        print(f"오류: 데이터셋 디렉토리를 찾을 수 없습니다: {dataset_dir}")
        return
    
    for filename in filenames:
        file_path = os.path.join(dataset_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"경고: {file_path}를 찾을 수 없습니다. 건너뜁니다.")
            continue
        
        try:
            print(f"--- CSV 전처리 시작: {file_path} ---")
            
            # CSV 파일 로드 및 검증
            df = pd.read_csv(file_path)
            validate_dataframe(df, ['input'])
            
            original_count = len(df)
            print(f"전처리 전: {original_count:,}개 행")
            
            # 전처리 적용
            df["input"] = df["input"].progress_apply(preprocess)
            
            # 빈 값 제거
            df = df.dropna(subset=["input"])
            df = df[df["input"].str.len() > 0]
            
            final_count = len(df)
            print(f"전처리 후: {final_count:,}개 행 (제거된 행: {original_count - final_count:,}개)")
            
            # 결과 저장
            df.to_csv(file_path, index=False, encoding="utf-8")
            print(f"--- 전처리 완료 및 저장: {file_path} ---")
            
        except Exception as e:
            print(f"오류: {filename} 처리 중 문제가 발생했습니다: {e}")
            continue
    
    print("모든 파일 처리 완료")


if __name__ == "__main__":
    main()